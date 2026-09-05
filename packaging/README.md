# Packaging Simplex for Windows

Four ways a person can end up with Simplex, in order of how much work each
costs you and how little it costs them.

| route | what the user does | what you maintain |
|-------|--------------------|-------------------|
| zip   | unzip, double-click `start.bat` | nothing |
| installer | run `Simplex-x.y.z-setup.exe`, click Next | one Inno script |
| winget | `winget install <you>.Simplex` | three YAML files per release |
| Scoop | `scoop install simplex` | one JSON per release |

All four end at the same place: the first launch opens the setup page in the
browser (`tools/setup_web.py`), which does the rest.

## Build

    powershell -ExecutionPolicy Bypass -File packaging\build.ps1 -Version 1.0.0 -Zip

Needs [Inno Setup 6](https://jrsoftware.org/isdl.php) for the installer; the
zip needs nothing. Output lands in `dist\`, and the script prints the SHA-256
of each artifact, which is what the manifests below want.

## Installer notes

`packaging/simplex.iss` installs **per user** into
`%LOCALAPPDATA%\Programs\Simplex` and asks for no elevation. That is
deliberate: Simplex writes into its own folder forever after - the virtualenv,
about ten gigabytes of weights, the conversation history - and a Program Files
install would mean either a UAC prompt on every launch or a program that
cannot write to itself.

The user can still change the folder on the wizard's directory page, which
matters when the system drive has no room for the weights.

Uninstall asks whether to keep the model, the settings and the conversations,
and keeps them by default, because most uninstalls are really reinstalls.

## Code signing

Unsigned, SmartScreen shows "Windows protected your PC" with a *More info*
link the first time anyone runs the installer. That warning fades as a
signed-and-downloaded reputation builds, and never fully goes away unsigned.

An EV or OV certificate (roughly $200-400/year) or
[Azure Trusted Signing](https://learn.microsoft.com/azure/trusted-signing/)
removes it. Sign both the installer and anything executable inside it:

    signtool sign /tr http://timestamp.digicert.com /td sha256 /fd sha256 ^
        /a dist\Simplex-1.0.0-setup.exe

If you are not signing, the honest thing is to say so on the download page and
publish the SHA-256, so people can check what they got.

## winget

Fill in every `CHANGEME` in `packaging/winget/*.yaml` - the publisher half of
the identifier has to match a domain or GitHub account you control - then:

    winget validate --manifest packaging\winget
    winget install  --manifest packaging\winget      # local install test

and open a pull request against
[microsoft/winget-pkgs](https://github.com/microsoft/winget-pkgs). The
`ProductCode` in the installer manifest is the Inno `AppId` with `_is1`
appended; keep them in step or upgrades will not be detected.

## Scoop

`packaging/scoop/simplex.json` points at the zip. The `persist` list is the
important part: `models`, `.venv`, `sessions` and `.env` are moved out of the
versioned folder, so an update does not throw away a ten gigabyte download.

Publish it in a bucket repository of your own; `scoop bucket add <name> <url>`
is what users then run.

## What is deliberately not here

**A single-file `Simplex.exe`.** PyInstaller can wrap the launcher, but the
result is unsigned, ~40 MB, and triggers *more* antivirus noise than a `.bat`,
while removing none of the real work (the venv, the engine, the weights still
have to arrive). The installer gives the same double-click experience with a
fraction of the trouble. If you do want one later, the entry point is
`tools/win_start.py:main` and nothing in the kit assumes it is run as a script.
