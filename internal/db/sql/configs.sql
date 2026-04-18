-- name: CreateConfig :execresult
INSERT INTO configs (key, value, description)
VALUES (?, ?, ?)
ON CONFLICT(key) DO UPDATE SET
    value = excluded.value,
    description = excluded.description,
    updated_at = datetime('now');

-- name: GetConfig :one
SELECT * FROM configs WHERE key = ?;

-- name: GetConfigValue :one
SELECT value FROM configs WHERE key = ?;

-- name: UpdateConfig :exec
UPDATE configs
SET value = ?,
    description = ?,
    updated_at = datetime('now')
WHERE key = ?;

-- name: DeleteConfig :exec
DELETE FROM configs WHERE key = ?;

-- name: ListConfigs :many
SELECT * FROM configs
ORDER BY key ASC;

-- name: ConfigExists :one
SELECT EXISTS(SELECT 1 FROM configs WHERE key = ?);

-- name: CreateConfigIfNotExists :exec
INSERT INTO configs (key, value, description)
VALUES (?, ?, ?)
ON CONFLICT(key) DO NOTHING;
