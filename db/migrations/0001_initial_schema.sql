-- 0001_initial_schema.sql
-- Initial schema for litter-detection Supabase backend.

create extension if not exists "pgcrypto";

-- ─── cameras ───────────────────────────────────────────────────────────
create table public.cameras (
  id          uuid primary key default gen_random_uuid(),
  name        text not null unique,
  rtsp_url    text,
  location    text,
  is_active   boolean not null default true,
  created_at  timestamptz not null default now()
);

-- ─── snapshots ─────────────────────────────────────────────────────────
create table public.snapshots (
  id               uuid primary key default gen_random_uuid(),
  camera_id        uuid references public.cameras(id) on delete set null,
  storage_path     text not null,
  captured_at      timestamptz not null,
  width            integer,
  height           integer,
  file_size_bytes  bigint,
  created_at       timestamptz not null default now()
);
create index snapshots_captured_at_idx on public.snapshots (captured_at desc);

-- ─── incidents (dump events) ───────────────────────────────────────────
create table public.incidents (
  id                uuid primary key default gen_random_uuid(),
  camera_id         uuid references public.cameras(id) on delete set null,
  snapshot_id       uuid references public.snapshots(id) on delete set null,
  occurred_at       timestamptz not null,
  litter_label      text not null,
  litter_confidence numeric(4,3) not null,
  person_confidence numeric(4,3) not null,
  track_id          integer,
  metadata          jsonb not null default '{}'::jsonb,
  created_at        timestamptz not null default now()
);
create index incidents_occurred_at_idx on public.incidents (occurred_at desc);
create index incidents_camera_id_idx   on public.incidents (camera_id);

-- ─── recordings ────────────────────────────────────────────────────────
create table public.recordings (
  id                uuid primary key default gen_random_uuid(),
  camera_id         uuid references public.cameras(id) on delete set null,
  storage_path      text not null,
  started_at        timestamptz not null,
  ended_at          timestamptz,
  duration_seconds  numeric,
  width             integer,
  height            integer,
  fps               numeric,
  file_size_bytes   bigint,
  created_at        timestamptz not null default now()
);
create index recordings_started_at_idx on public.recordings (started_at desc);

-- ─── alerts ────────────────────────────────────────────────────────────
create table public.alerts (
  id            uuid primary key default gen_random_uuid(),
  incident_id   uuid not null references public.incidents(id) on delete cascade,
  channel       text not null check (channel in ('email','sms')),
  recipient     text,
  status        text not null check (status in ('sent','failed','skipped')),
  error_message text,
  sent_at       timestamptz not null default now()
);
create index alerts_incident_id_idx on public.alerts (incident_id);

-- ─── storage buckets ───────────────────────────────────────────────────
insert into storage.buckets (id, name, public)
values
  ('snapshots',  'snapshots',  true),
  ('recordings', 'recordings', true)
on conflict (id) do nothing;
