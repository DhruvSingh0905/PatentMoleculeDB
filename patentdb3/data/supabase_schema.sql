-- patentdb3 -> Supabase. ONE joined table.
--
-- The pipeline's normalised shape — compounds, assays, measurements — mirrors
-- how the EXTRACTOR works, not how anyone reads the result. Three tables and
-- two views to answer "what did this patent measure" is the wrong deliverable.
-- Those stay local. What ships is the joined set: one row per compound with
-- its assays nested.
--
-- WHY JSONB AND NOT A COLUMN PER ASSAY. 553 distinct assay headings over 137
-- patents, and a new one with every patent added. The heading is DATA, not
-- schema. What keeps it queryable is that the axes are lifted out of each
-- heading — `metric`, `target`, and a value converted to one unit — so a
-- question crosses documents that spell the assay differently:
--
--     where assays @> '[{"metric":"IC50"}]'
--     where 'IC50' = any(metrics)          -- flat arrays for the common case
--     where best_um < 1.0                  -- lifted out, so sorting is free
--
-- The patent's own words survive verbatim in each object's `assay`.
--
-- EACH OBJECT CARRIES ONLY THE KEYS ITS HEADING FILLED. The seventeen possible
-- fields are the union over every heading, and no heading uses them all — a
-- letter grade has no value and no unit, a concentration has no dose and no
-- species. Writing the whole set made 57.3% of every key null, so an object
-- now averages 7 keys instead of 17. A missing key is not missing data:
-- `a->>'grade' is null` is true for an absent key and a null one alike, and
-- `@>` tests only the keys it is given.

create table compounds (
    patent_id      text not null,
    -- Kept because a later join needs it and nothing else stands in for it.
    -- ABSENT from `compound_data`, the reading surface: the patent's own
    -- number tells a reader nothing, and two patents both numbering a
    -- compound `7` invites exactly the wrong comparison.
    cid            text not null,
    name           text,
    smiles         text,
    inchikey       text,

    -- WHERE THE STRUCTURE CAME FROM. Not equivalent, and a reader filtering
    -- on quality wants this before anything else.
    --   xml        read from the document's text; OPSIN parsed the name
    --   molscribe  the patent DREW it and named it nowhere
    --   markush    a scaffold plus substituent columns
    --   gp         a Google Patents annotation: no compound number, no assays
    route          text,
    -- Whether it is finished, in plain english. A drawn row is not a failure —
    -- it is work queued behind image recognition, and it says so.
    status         text,
    -- The drawing, for rows that have one and no structure yet. Carrying the
    -- filename is what lets someone fill these in later without going back to
    -- the XML to find out which picture to read.
    image_file     text,
    image_ref      text,

    -- The mass the document prints for this compound, and whether our
    -- structure agrees with it. The only correctness signal here that needs
    -- no external reference.
    reported_mz    numeric,
    mass_check     text,
    mass_delta     numeric,

    n_assays       integer not null default 0,
    n_measurements integer not null default 0,
    best_um        numeric,
    metrics        text[],
    targets        text[],
    assays         jsonb   not null default '[]'::jsonb,
    primary key (patent_id, cid)
);

create index ix_cmp_route   on compounds (route);
create index ix_cmp_status  on compounds (status);
create index ix_cmp_key     on compounds (inchikey);
create index ix_cmp_best    on compounds (best_um);
create index ix_cmp_assays  on compounds using gin (assays);
create index ix_cmp_metrics on compounds using gin (metrics);
create index ix_cmp_targets on compounds using gin (targets);

-- Reference data, not user data: every row is already public in a granted
-- patent, and nothing writes through the API. A load opens an INSERT policy
-- and drops it again; the resting state is read-only.
alter table compounds enable row level security;
create policy "read compounds" on compounds for select using (true);

-- THE READING SURFACE. Everything, minus the patent's internal number.
create or replace view compound_data as
select patent_id, name, smiles, inchikey, route, status,
       image_file, reported_mz, mass_check,
       n_assays, n_measurements, best_um, metrics, targets, assays
  from compounds;
alter view compound_data set (security_invoker = true);
