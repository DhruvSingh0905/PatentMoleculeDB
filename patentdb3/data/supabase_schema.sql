-- patentdb3 -> Supabase. The staging schema.
--
-- THE PROBLEM THIS SHAPE SOLVES. Every patent names its own assays. There are
-- 553 distinct assay headings over 137 documents and almost none of them
-- agree: `PI3K alpha IC50 (nM)`, `FLAP Binding wild type HTRF Ki (uM)`,
-- `% inhibition of IL6`, `Septoria Rating`. A wide table would need a column
-- per heading and would grow one every time a patent is added.
--
-- So the measurement table is LONG — one row per value — and the heading is
-- data, not schema. What makes it queryable anyway is that the AXES are lifted
-- out of the heading into their own columns:
--
--     assays.metric      IC50 / Ki / EC50 / Kd / percent      parsed
--     assays.target_raw  the target token the patent used     parsed
--     measurements.value_um    every value in one unit        converted
--
-- That is the whole trick. Ask for "every IC50 against BTK under 1 uM" across
-- all 137 patents and it is one WHERE clause, even though no two documents
-- spell the assay the same way. The patent's own words survive verbatim in
-- `assays.assay_name`, so nothing is lost to the normalisation.
--
-- Units are micromolar because that is the unit the deliverable asks for. The
-- number the patent actually printed stays beside it in `value_numeric` and
-- `unit`, so a reader can always get back to the source.

create table if not exists compounds (
    patent_id    text    not null,
    cid          text    not null,           -- the patent's own compound number
    name         text,                       -- IUPAC, as the document states it
    smiles       text,
    inchikey     text,
    source       text,                       -- cid_first | table | heading
    -- What the document itself says this compound weighs, and whether our
    -- structure agrees. `mass_check` is the only correctness signal here that
    -- needs no external reference.
    reported_mz  numeric,
    mass_check   text,                       -- agrees | contradicts | (blank)
    mass_delta   numeric,
    markush      boolean not null default false,
    drawn_only   boolean not null default false,
    primary key (patent_id, cid)
);

-- One row per distinct heading per patent. This is where a heading stops being
-- free text and becomes something you can filter on.
create table if not exists assays (
    assay_id    bigint generated always as identity primary key,
    patent_id   text not null,
    assay_name  text not null,               -- VERBATIM. The patent's own words.
    metric      text,                        -- IC50 | EC50 | Ki | Kd | percent
    target_raw  text,                        -- the target token, as written
    unit        text,                        -- the unit the patent printed
    table_id    text,
    unique (patent_id, assay_name, table_id)
);

-- The long table. One row per measurement.
create table if not exists measurements (
    id            bigint generated always as identity primary key,
    patent_id     text    not null,
    cid           text    not null,
    assay_id      bigint  not null references assays (assay_id),
    -- as the patent printed it
    value_numeric numeric,
    qualifier     text,                      -- < > = ~ etc
    unit          text,
    -- normalised, so two patents can be compared
    value_um      numeric,
    -- a graded value carries an interval instead of a number
    letter_grade  text,
    range_lo_um   numeric,
    range_hi_um   numeric,
    n_runs        integer,
    table_id      text,
    column_header text,
    foreign key (patent_id, cid) references compounds (patent_id, cid)
);

-- Structures from Google Patents' own annotations. NO compound number by
-- construction, so these join to nothing and live apart — extra molecules the
-- document mentions, not measurements.
create table if not exists gp_compounds (
    id        bigint generated always as identity primary key,
    patent_id text not null,
    gp_name   text,
    smiles    text,
    inchikey  text,
    label     text,                           -- compound | reagent | fragment
    finished  boolean not null default false, -- survived the conservative filter
    unique (patent_id, inchikey)
);

create index if not exists ix_meas_compound on measurements (patent_id, cid);
create index if not exists ix_meas_assay    on measurements (assay_id);
create index if not exists ix_meas_value    on measurements (value_um);
create index if not exists ix_assays_metric on assays (metric);
create index if not exists ix_assays_target on assays (target_raw);
create index if not exists ix_cmp_inchikey  on compounds (inchikey);
create index if not exists ix_gp_inchikey   on gp_compounds (inchikey);

-- The join everyone actually wants: one row per measurement with its compound
-- and its assay already attached.
create or replace view v_measurements as
select m.patent_id,
       m.cid,
       c.name,
       c.smiles,
       c.inchikey,
       a.assay_name,
       a.metric,
       a.target_raw,
       m.qualifier,
       m.value_um,
       m.unit          as printed_unit,
       m.value_numeric as printed_value,
       m.letter_grade,
       m.range_lo_um,
       m.range_hi_um,
       m.n_runs,
       c.reported_mz,
       c.mass_check,
       m.table_id
  from measurements m
  join assays    a on a.assay_id = m.assay_id
  join compounds c on c.patent_id = m.patent_id and c.cid = m.cid;

-- Read-only for anyone with the anon key. This is reference data, not user
-- data: nobody writes to it through the API, and every row is already public
-- in a granted patent.
alter table compounds     enable row level security;
alter table assays        enable row level security;
alter table measurements  enable row level security;
alter table gp_compounds  enable row level security;

create policy "read compounds"    on compounds    for select using (true);
create policy "read assays"       on assays       for select using (true);
create policy "read measurements" on measurements for select using (true);
create policy "read gp_compounds" on gp_compounds for select using (true);

-- ─────────────────────────────────────────────────────────────────────────
-- ONE ROW PER COMPOUND. This is the table to read.
--
-- `v_measurements` is one row per MEASUREMENT, so a compound with nine assays
-- appears nine times with its name and SMILES repeated. Right for aggregating,
-- wrong for reading, and not what anyone wants to open.
--
-- A column per assay is impossible — 553 distinct headings, and a new one with
-- every patent added. So the assays collapse into a JSONB array on the
-- compound's own row: one object per measurement, each carrying the heading
-- verbatim plus the parsed metric and target and the normalised value.
--
-- JSONB rather than text because it stays QUERYABLE, and the GIN indexes make
-- it cheap:
--
--     where assays @> '[{"metric": "IC50"}]'
--     where 'IC50' = any(metrics)          -- flat arrays for the common case
--     where best_um < 1.0                  -- lifted out so sorting is free
--
-- REFRESH IT after any reload: `refresh materialized view compound_profiles;`
-- It is materialised rather than a plain view because it aggregates 119,306
-- measurements and is read far more often than the data changes.
--
-- The linter warns that a materialized view is readable by `anon` and sits
-- outside row level security. That is accepted here rather than missed: every
-- base table's policy is already `for select using (true)`, so this exposes
-- exactly the data it is meant to and nothing further.
create materialized view if not exists compound_profiles as
select c.patent_id, c.cid, c.name, c.smiles, c.inchikey,
       c.reported_mz, c.mass_check, c.markush,
       count(m.id)                                         as n_measurements,
       count(distinct a.assay_name)                        as n_assays,
       min(m.value_um)                                     as best_um,
       array_remove(array_agg(distinct a.metric), null)     as metrics,
       array_remove(array_agg(distinct a.target_raw), null) as targets,
       coalesce(jsonb_agg(jsonb_build_object(
           'assay', a.assay_name, 'metric', a.metric, 'target', a.target_raw,
           'qualifier', m.qualifier, 'value_um', m.value_um,
           'value', m.value_numeric, 'unit', m.unit,
           'grade', m.letter_grade, 'lo_um', m.range_lo_um,
           'hi_um', m.range_hi_um, 'n_runs', m.n_runs, 'table', m.table_id)
         order by a.assay_name) filter (where m.id is not null),
         '[]'::jsonb)                                      as assays
  from compounds c
  left join measurements m on m.patent_id = c.patent_id and m.cid = c.cid
  left join assays a on a.assay_id = m.assay_id
 group by c.patent_id, c.cid, c.name, c.smiles, c.inchikey,
          c.reported_mz, c.mass_check, c.markush;

create unique index if not exists ix_prof_pk      on compound_profiles (patent_id, cid);
create index if not exists ix_prof_assays  on compound_profiles using gin (assays);
create index if not exists ix_prof_metrics on compound_profiles using gin (metrics);
create index if not exists ix_prof_targets on compound_profiles using gin (targets);
create index if not exists ix_prof_best    on compound_profiles (best_um);
create index if not exists ix_prof_key     on compound_profiles (inchikey);
