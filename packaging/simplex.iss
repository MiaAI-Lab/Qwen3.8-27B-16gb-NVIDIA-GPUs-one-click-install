; Inno Setup script for Simplex.  Build with:  iscc packaging\simplex.iss
;
; Per-user by design.  Simplex writes into its own folder for the rest of its
; life - the virtualenv, ten gigabytes of weights, the conversation history -
; so installing into Program Files would mean either a UAC prompt on every
; launch or a program that cannot write to itself.  {localappdata}\Programs is
; the location Windows intends for exactly this, and it needs no elevation.

#define AppName        "Simplex"
; overridable from the command line: ISCC /DAppVersion=1.1.0
#ifndef AppVersion
  #define AppVersion   "1.0.0"
#endif
#define AppPublisher   "Simplex"
#define AppURL         "https://github.com/CHANGEME/simplex"
#define AppExeName     "start.bat"
#define SourceDir      ".."

[Setup]
AppId={{8E1C6A4E-6F1B-4C2E-9A7D-5B3F0E2A91C4}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher={#AppPublisher}
AppSupportURL={#AppURL}
AppUpdatesURL={#AppURL}
DefaultDirName={localappdata}\Programs\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
DisableDirPage=no
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
OutputDir=..\dist
OutputBaseFilename=Simplex-{#AppVersion}-setup
SetupIconFile=..\tools\simplex.ico
UninstallDisplayIcon={app}\tools\simplex.ico
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
; "x64compatible" would be more precise but needs Inno Setup 6.3+; "x64" is
; accepted by every 6.x, and this kit needs an NVIDIA GPU anyway.
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64
; The installer is small; the weights are downloaded on first run. Declare the
; space that download will actually need so the wizard does not promise 8 MB
; and then ask for ten gigabytes.
ExtraDiskSpaceRequired=10737418240
DisableWelcomePage=no

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; \
  GroupDescription: "Shortcuts:"
Name: "startup"; Description: "Start Simplex when I sign in"; \
  GroupDescription: "Shortcuts:"; Flags: unchecked

[Files]
; Everything the kit is made of.  Excluded: anything the kit creates for
; itself, which must survive an upgrade and must never ship in an installer.
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs; \
  Excludes: "\.venv,\.git,\.github,\models,\sessions,\workspace,\logs,\dist,\build,\.simplex,\.env,\providers.json,\bench_vram.json,*.pyc,__pycache__,*.log"

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExeName}"; \
  WorkingDir: "{app}"; IconFilename: "{app}\tools\simplex.ico"; \
  Comment: "Chat with a local model on your own GPU"
Name: "{group}\Simplex folder"; Filename: "{app}"
Name: "{group}\Uninstall {#AppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; \
  WorkingDir: "{app}"; IconFilename: "{app}\tools\simplex.ico"; Tasks: desktopicon
; runminimized to match the shortcuts tools/shortcuts.py writes - a console
; window taking focus at every sign-in is not what "start with Windows" means
Name: "{userstartup}\{#AppName}"; Filename: "{app}\{#AppExeName}"; \
  WorkingDir: "{app}"; IconFilename: "{app}\tools\simplex.ico"; \
  Flags: runminimized; Tasks: startup

[Run]
Filename: "{app}\{#AppExeName}"; Description: "Set up and start Simplex now"; \
  WorkingDir: "{app}"; Flags: postinstall nowait skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}\__pycache__"
Type: filesandordirs; Name: "{app}\tools\__pycache__"
Type: filesandordirs; Name: "{app}\.simplex"

[Code]
var
  PythonWarned: Boolean;

function PythonPresent(): Boolean;
var
  Code: Integer;
begin
  { "py -3 --version" is the same test start.bat makes. }
  Result := Exec(ExpandConstant('{cmd}'), '/c py -3 --version >nul 2>nul || python --version >nul 2>nul',
                 '', SW_HIDE, ewWaitUntilTerminated, Code) and (Code = 0);
end;

function InitializeSetup(): Boolean;
var
  ErrorCode: Integer;
begin
  Result := True;
  PythonWarned := False;
  if not PythonPresent() then
  begin
    PythonWarned := True;
    if MsgBox('Simplex needs Python 3.11 or newer, and it is not installed on this PC.' + #13#10#13#10 +
              'Install Simplex anyway? You will need to install Python before the first launch.' + #13#10 +
              'Choosing No opens the Python download page instead.',
              mbConfirmation, MB_YESNO) = IDNO then
    begin
      ShellExec('open', 'https://www.python.org/downloads/', '', '', SW_SHOW, ewNoWait, ErrorCode);
      Result := False;
    end;
  end;
end;

function ReadEnvModelDir(): String;
var
  Lines: TArrayOfString;
  I: Integer;
  Line: String;
begin
  Result := '';
  if not LoadStringsFromFile(ExpandConstant('{app}\.env'), Lines) then
    exit;
  for I := 0 to GetArrayLength(Lines) - 1 do
  begin
    Line := Trim(Lines[I]);
    if (Pos('MODEL_DIR=', Line) = 1) then
      Result := Trim(Copy(Line, Length('MODEL_DIR=') + 1, Length(Line)));
  end;
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  Keep: Integer;
  ModelDir: String;
begin
  if CurUninstallStep = usPostUninstall then
  begin
    { The virtualenv, the weights and the conversations are the expensive
      things here, and a person uninstalling to reinstall wants them kept. }
    Keep := MsgBox('Keep the downloaded model, your settings and your conversations?' + #13#10#13#10 +
                   'Yes  - leave the models, .env, and sessions folders in place (several GB)' + #13#10 +
                   'No   - delete everything Simplex created',
                   mbConfirmation, MB_YESNO);
    if Keep = IDNO then
    begin
      DelTree(ExpandConstant('{app}\.venv'), True, True, True);
      { MODEL_DIR may point somewhere else entirely; .env is the only record
        of where, and it is about to be deleted, so read it first. }
      ModelDir := ReadEnvModelDir();
      if (ModelDir <> '') and (Pos(':', ModelDir) > 0) then
        DelTree(ModelDir, True, True, True);
      DelTree(ExpandConstant('{app}\models'), True, True, True);
      DelTree(ExpandConstant('{app}\sessions'), True, True, True);
      DelTree(ExpandConstant('{app}\workspace'), True, True, True);
      DelTree(ExpandConstant('{app}\logs'), True, True, True);
      DeleteFile(ExpandConstant('{app}\.env'));
      DeleteFile(ExpandConstant('{app}\providers.json'));
      RemoveDir(ExpandConstant('{app}'));
    end;
  end;
end;
