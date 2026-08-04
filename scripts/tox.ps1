param(
    [ValidateSet("all", "lint", "typecheck")]
    $Target = "all"
)

switch ($Target) {
    "all" { tox }
    "lint" { tox -e lint }
    "typecheck" { tox -e typecheck }
}
