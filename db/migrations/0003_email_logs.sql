-- 0003_email_logs.sql
-- Store outbound email alert delivery attempts.

create table if not exists public.email_logs (
  id               uuid primary key default gen_random_uuid(),
  camera_id        uuid references public.cameras(id) on delete set null,
  occurred_at      timestamptz not null default now(),
  litter_label     text,
  recipients       text[] not null default '{}'::text[],
  status           text not null check (status in ('sent', 'failed', 'skipped')),
  provider         text not null default 'resend',
  provider_message_id text,
  error_message    text
);

create index if not exists email_logs_occurred_at_idx on public.email_logs (occurred_at desc);
create index if not exists email_logs_camera_id_idx on public.email_logs (camera_id);
