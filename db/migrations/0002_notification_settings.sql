-- 0002_notification_settings.sql
-- Persist app-level notification settings per camera.

create table if not exists public.notification_settings (
  camera_id         uuid primary key references public.cameras(id) on delete cascade,
  email_recipients  text[] not null default '{}'::text[],
  updated_at        timestamptz not null default now()
);
