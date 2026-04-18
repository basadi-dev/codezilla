-- name: CreateSession :execresult
INSERT INTO sessions (id, title, model, working_directory, metadata)
VALUES (?, ?, ?, ?, ?);

-- name: GetSession :one
SELECT * FROM sessions WHERE id = ?;

-- name: UpdateSession :exec
UPDATE sessions
SET title = ?,
    status = ?,
    model = ?,
    total_tokens = ?,
    total_requests = ?,
    working_directory = ?,
    metadata = ?,
    updated_at = datetime('now')
WHERE id = ?;

-- name: UpdateSessionStatus :exec
UPDATE sessions
SET status = ?,
    updated_at = datetime('now')
WHERE id = ?;

-- name: DeleteSession :exec
DELETE FROM sessions WHERE id = ?;

-- name: ListSessions :many
SELECT * FROM sessions
ORDER BY updated_at DESC
LIMIT ? OFFSET ?;

-- name: ListActiveSessions :many
SELECT * FROM sessions
WHERE status = 'active'
ORDER BY updated_at DESC
LIMIT ? OFFSET ?;

-- name: SearchSessions :many
SELECT * FROM sessions
WHERE title LIKE ? OR metadata LIKE ?
ORDER BY updated_at DESC
LIMIT ? OFFSET ?;

-- name: CountSessions :one
SELECT COUNT(*) FROM sessions;

-- name: CountActiveSessions :one
SELECT COUNT(*) FROM sessions WHERE status = 'active';

-- name: GetSessionStats :one
SELECT 
    COUNT(*) as total_sessions,
    SUM(total_tokens) as total_tokens,
    SUM(total_requests) as total_requests
FROM sessions;

-- name: AddMessage :execresult
INSERT INTO messages (session_id, role, content, tool_name, tool_args, tool_result, token_count)
VALUES (?, ?, ?, ?, ?, ?, ?);

-- name: GetMessagesBySession :many
SELECT * FROM messages
WHERE session_id = ?
ORDER BY created_at ASC;

-- name: GetMessagesBySessionWithLimit :many
SELECT * FROM (
    SELECT * FROM messages
    WHERE session_id = ?
    ORDER BY created_at DESC
    LIMIT ?
) ORDER BY created_at ASC;

-- name: GetMessage :one
SELECT * FROM messages WHERE id = ?;

-- name: DeleteMessagesBySession :exec
DELETE FROM messages WHERE session_id = ?;

-- name: CountMessagesBySession :one
SELECT COUNT(*) FROM messages WHERE session_id = ?;

-- name: GetRecentSessions :many
SELECT * FROM sessions
ORDER BY updated_at DESC
LIMIT ?;

-- name: ArchiveOldSessions :exec
UPDATE sessions
SET status = 'archived'
WHERE status = 'active'
  AND updated_at < datetime('now', ?);
