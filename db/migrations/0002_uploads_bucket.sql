-- 0002_uploads_bucket.sql
-- Public bucket for user-uploaded test videos.

insert into storage.buckets (id, name, public)
values ('uploads', 'uploads', true)
on conflict (id) do nothing;
