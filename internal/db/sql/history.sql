-- name: AddHistory :exec
INSERT INTO prompt_history (prompt, session_id) 
VALUES (?, ?)
ON CONFLICT DO NOTHING;

-- name: GetRecentHistory :many
SELECT prompt FROM prompt_history 
ORDER BY id DESC 
LIMIT ?;

-- name: SearchHistory :many
SELECT prompt FROM prompt_history 
WHERE prompt LIKE '%' || ? || '%' 
ORDER BY id DESC;

-- name: ClearHistory :exec
DELETE FROM prompt_history;
