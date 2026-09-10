# Security Policy

## Supported versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a vulnerability

Please do not open a public issue for security problems.

Use GitHub's private vulnerability reporting for this repository:
<https://github.com/DiogoRibeiro7/drl-cox/security/advisories/new>.
If that is not available to you, email the maintainer at <dfr@esmad.ipp.pt>
with a description of the issue, steps to reproduce, and the affected version.

You can expect an acknowledgement within seven days. Fixes are released as
patch versions and credited in the changelog unless you prefer to stay
anonymous.

## Scope

`drl-cox` is a numerical library. It does not open network connections or
execute untrusted code, but it does read CSV files passed to
`load_whas500_like_csv`; treat inputs from untrusted sources with the usual
care.
