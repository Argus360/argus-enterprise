# Changelog

What changed in each release, written for the person who has to decide whether
to update. Dates are release dates. Every version is built by CI from a tag on
`main` and published with SHA-256 sums — verify your download against them.

## v1.10.1 — 12 Oct 2026

Preview fixes from the pilot server's own diagnostics, by Demi.

**The camera wall costs far less when monitoring is stopped.** Every tile
was being encoded at the camera's full resolution, up to 2592 x 1944, eight
times a second, to keep the zone editor's snapshot sharp. On the pilot
server that kept the preview process busy with nothing running. Tiles are
now encoded at 960 pixels wide; the zone editor still gets a full-resolution
snapshot when it asks for one.

**A stopped preview is never reused.** Closing a tile while its camera was
not answering could leave a half-stopped capture behind, and the next time
the tile opened it got that one instead of a fresh connection. Those tiles
sat on "waiting for the first image" indefinitely. The capture is now
replaced.

**Support bundles say what each preview is doing.** Running, stopping,
waiting for an image, or never started, and whether its reader thread is
still alive, so a camera that never answers is told apart from a stuck
preview.

## v1.10.0 — 10 Oct 2026

**Choose where alerts are checked.** Until now the AI that double-checks
every alert ran on the computer Argus is installed on. On a server without
a graphics card that took a minute or more per alert, and the check often
gave up. Settings → AI verification (and the AI model step of setup) now
offers a choice: on this computer, or a cloud provider — OpenRouter, Groq,
Google Gemini, Anthropic Claude, or your own OpenAI-compatible endpoint.
Paste the provider's API key, press Test, and the next Start monitoring
sends alert frames there over HTTPS and gets a verdict in seconds. Live
video never leaves the site; only the few still images behind a candidate
alert do, and nothing at all for alerts the rules confirm on their own.

**The key is kept privately.** It is stored in a file only your account can
read, beside the login database, and handed to the engine through its
environment. It is never written into the site configuration, never shown
in a process list, and never included in a support bundle.

**Cloud models now see the whole alert.** The generic cloud path used to
send only the first frame, so the model never saw the zoomed-in subject
crop that decides most verdicts. Every frame goes now, with retries when a
provider is busy.

**Scene mapping, English rules and watches follow the choice.** Choosing a
cloud verifier used to send scene mapping back to the on-device model and
switched English rules and watches off entirely. All three now run on the
same provider as alert verification.

## v1.9.5 — 10 Oct 2026

Pilot wall fixes from the server logs, by Demi.

**Saving a camera keeps its place.** Changing a camera's detector settings
used to move it to the end of the list, which reshuffled the wall and
re-subscribed tiles that were playing fine. Cameras now stay in order.

**The pre-monitoring preview no longer runs behind.** The preview used to
sleep after every frame it read, so a camera sending 30 frames a second
built up a backlog and the picture drifted seconds behind live. The preview
now drains the camera at its own pace and only throttles how many pictures
it encodes.

**A stream that never shows its first picture gives up and retries.** A tile
could sit on "Connecting" indefinitely. It now times out after 20 seconds and
goes through the normal retry path.

**Person boxes are never seconds old.** On a slow box the smooth preview
could paint boxes from an earlier frame onto newer video. Boxes now expire
one second after the frame they belong to; a slow detector means fewer boxes,
not wrong ones.

**Support bundles carry preview diagnostics.** The bundle now records how
each preview stream was opened, how long the first picture took, and whether
the saved health files were fresh when the bundle was made.

## v1.9.4 — 8 Oct 2026

**Stopping monitoring no longer leaves a video relay running.** On Windows,
each Stop monitoring left behind the relay that carries the camera streams,
still connected to the cameras. After a few restarts the pilot server had
three of them, each holding connections that cameras which allow only a
couple of viewers needed for the next start. The relay is now shut down when
monitoring stops, and any left over from an earlier version is cleaned up
before a new one starts.

## v1.9.3 — 7 Oct 2026

Fixes found on the first day on the pilot's own server.

**Test connection tells the truth.** A camera whose stream could not be
opened was reported as reachable, so cameras were saved that had never
produced a picture. The test now succeeds only when a video frame has
actually been decoded, and says so with the frame size: *Video received
1920 x 1080*. Anything less is reported as the failure it is.

**Camera names with a slash work.** A camera called *SMB BAY 3/4* could be
added but then nothing else worked for it: no preview, no delete. The slash
in its name broke the address the app used behind the scenes. Fixed without
renaming anything already saved.

**Starting is shown as starting.** While the engine loads its models the
header no longer flips between "running" and "stopped"; it says Starting
until the first frames arrive. Network cameras that do not answer now time
out in eight seconds instead of hanging the start.

**Normal movement and Multiple people moving are back** in each camera's
Detectors list; the group had gone missing from the screen.

**Settings has a Diagnostics export** for support: logs and health with
passwords, camera addresses and tokens blanked out. Nothing is sent anywhere
on its own.

## v1.9.2 — 5 Oct 2026

**PPE checks run whether or not anyone is watching the cameras.** The PPE
scanner only received the people in frame while the live picture was being
published. Headless runs, including the automated checks, handed it empty
frames: a yard full of workers with no hard hats and never a violation. The
people now reach the scanner in every mode. Found running the pilot's KPI set
end to end, where the same clip then produced the missing-hard-hat alerts it
should. This is the build for the pilot: every change from today's work is in it.

## v1.9.1 — 5 Oct 2026

**Download AI installs the object-recognition model too.** Recognising a
customer's own products from reference photos needs a second model on the
machine, and until now a fresh install had the feature without its files, so
Object watchlists simply read "unavailable". The one Download AI click now
fetches both: the verification model as before, and the recognition model
(0.8 GB) beside it, resumed if the connection drops, checked file by file,
and loaded once to make sure it works before it is called ready. The AI
model step shows its progress on its own line. Finishing setup does not wait
for it: a site that never enrols a product loses nothing, and if it fails
only object recognition is off — cameras and monitoring are untouched.

**Pilot work from the CHI handoff.** Person tracking boxes can be shown per
camera or on every camera, and hiding them never stops detection. Possible
concealment alerts carry clearer evidence and wording, and repeat alerts for
the same person are held back. Registered objects can be drawn on a live
camera picture. A webcam preview waits for the camera's exposure to settle
before showing a picture, and a stopped engine hands its tiles back to the
preview instead of a stale frame.

## v1.9.0 — 4 Oct 2026

**Alerts by email.** Settings → Alerts has a new Email channel beside
Telegram and WhatsApp. Give it your mail server, a login and the addresses
to send to, and every alert arrives as one message: a header coloured by
priority, the alert in words, the reason the verifier gave, the camera, time
and confidence, the evidence pictures inline and the short clip attached
when it is small enough. Works with Gmail and Google Workspace (use an App
Password), Microsoft 365, and an internal relay with no login. Press *Send
test alert* after saving: if the password is wrong the app now says so,
instead of reporting a test that never left.

**One incident, one alert.** On a clip of an ATM being broken into, the
verifier confirmed fire, panic running, violence *and* theft — four alerts
for one event, three of them wrong. It had been asked whether something
bad was happening, and something bad was. It is now asked whether the
*specific* thing the detector claimed is in the picture, and a different
incident is a no: the right detector raises its own alert. The verifier's
wording changed, so its measured precision and recall are marked as owed
until the golden set is replayed.

## v1.8.25 — 2 Oct 2026

**Cameras show a live picture before monitoring starts.** Until now a camera
tile stayed "offline" until you pressed Start monitoring, because only the
detection engine could produce a stream. Watch now opens the camera itself
when monitoring is stopped and shows it labelled LIVE PREVIEW, and the zone
editor takes its snapshot from that same picture at the camera's real
resolution. The preview is handed over to the engine when you press Start
and comes back when you press Stop. Nothing is analysed while previewing, and
the picture never leaves this computer.

**Start monitoring says what it is waiting for.** If a camera is still being
let go of by the preview when you press Start (a network camera can take a
few seconds to answer), the app now says *Camera preview is still releasing
its capture; retry monitoring shortly* instead of failing with no words, and
the Watch tiles keep working in the meantime.

**A corrupt events database is quarantined on Windows too.** The start-up
integrity check moved a broken database aside so a fresh one could start, but
on Windows it still had the broken file open when it tried, so the move was
refused and the site stayed down with "corrupt". The file is closed first
now. Two configuration backups taken inside the same second no longer
collide on Windows either.

## v1.8.24 — 28 Sep 2026

**Vehicle entering and exiting is now something you draw.** Open a camera's
Zones tab, choose the gate-line tool, click where the line starts and where it
ends. An arrow across the line shows which way counts as *entering*; Swap
direction turns it round. A vehicle whose centre crosses the line one way is
an entry and the other way an exit, counted once per vehicle — a parked car
never crosses it. One line per camera; it is stored as a fraction of the
frame, so a change of camera resolution does not move it. Checked end to end:
a site built entirely through the app's own API, lines included, sent vehicle
entered / exited alerts to Telegram from recorded gate footage.

**Setup starts with the AI model, and Finish waits for it.** The 3.3 GB
on-device model used to sit in a Verification step near the end, where it was
easy to walk past — and a site finished without it had cameras whose scenes
were never read and alerts nobody checked. The download now starts on step 1
and carries on while locations and cameras are set up, with its progress shown
on every step. There is no skip. Finish is the one thing that waits for the
model, and the reason is printed above the button with the percentage, so it
is never a button that silently does nothing. Going offline is named as such,
and the download resumes where it stopped. Cameras is one step for as many
cameras as the site has, with Review scene and Draw zone beside each.

**Detector toggles show their real state.** Every toggle on a camera's
Detectors tab rendered off, whatever the engine had, and clicking one could
only ever switch it on. The camera list now carries each detector's actual
setting, so a toggle reads correctly and turns off as well as on.

**The Linux webcam message stops pointing at Windows.** A webcam that could
not be opened on Linux told the operator to open a Windows settings page. The
advice is now per operating system: on Linux, check that `/dev/video0` exists
and that your user is in the `video` group.

## v1.8.23 — 26 Sep 2026

**macOS no longer reports Argus as damaged.** Without an Apple certificate the
app was shipped with no signature at all, and Apple Silicon reports an
unsigned app as corrupt rather than unsigned — so a perfectly good download
said "Argus is damaged and can't be opened", and the only way past it was a
Terminal command. The app is now signed as it is packaged, so macOS shows the
ordinary "unidentified developer" prompt instead, which right-click → Open
clears. Full notarisation still needs an Apple developer account.

**The Windows installer is tested on every build.** Until now the build
checked the unpacked application; what you actually double-click — the
installer — had never been run anywhere. Each build now installs Argus
silently and starts that installed copy, creates an account, adds a camera and
waits for the engine to connect to it. A build that cannot do that is not
published.

**Fewer false theft alerts.** The video-action theft signal raised an alert on
one normal clip in five, and each one costs the on-device AI a verification it
could have spent on something real. It now needs to be far more certain before
it says anything.

**The audit log no longer holds its file open.** Windows will not let anything
move or delete a file a running program holds, which blocked upgrading or
uninstalling over an existing installation. The log also now writes each entry
in a single step, so two things happening at once can no longer break the
chain that proves the log has not been edited.

## v1.8.22 — 23 Sep 2026

Two things found live on a Windows screen share, and proof that the app starts.

**Test connection now works for a webcam.** Typing `0` for the built-in
camera and pressing Test connection always answered "Could not open", on
every computer — the test handed the camera index to the video library as
text, which it read as a file name. Adding the camera and starting monitoring
worked all along; now the test does too, and if the webcam is busy the
message says so: close the app using it (Meet, Teams, Zoom) and check the
camera permission for desktop apps.

**The Make list can be left empty, and every make can be chosen.** "Generic"
could not be selected (it shared a stream path with another entry and the
list kept showing that one), and there was no way to have no make at all. The
list now starts at "None — I'll type the address myself", which is the right
choice for a webcam or a pasted RTSP address.

**Every Windows build now starts the installed app itself** — not just its
engine — with a clean data directory: the app comes up, an owner is created,
a camera is added, the engine connects to it. A build that cannot do that
does not ship.

## v1.8.21 — 22 Sep 2026

Proof instead of guesswork on Windows.

**Fire and smoke alerts no longer fire on an ordinary room.** The fire
detector judged colour alone: a scene with a little warm colour, or a lot of
grey — every night-vision camera — was a fire candidate on every frame, so a
critical fire alert appeared seconds after every start and kept returning.
It now looks for warm or smoke-like regions that are *new* to the scene and
keep changing, learns the camera's own colours as background, ignores the
colour test on black-and-white night video, and reports at most one
candidate per camera every five minutes unless it grows. Measured on our
footage: an indoor camera went from a candidate on 39 of 39 frames to none;
a night driveway from 116 of 116 to none; real fires are still caught.

**Every Windows build now proves its replay clips play.** The shipped engine
writes one test clip during the build, and the build fails unless it comes
out as H.264 — the format the app's player decodes. Until now only a
Windows user could tell us whether the bundled converter ran.

**The AI runtime's own log is kept.** When the on-device model refuses to
load — not enough memory, a missing library, a damaged download — the
runtime says why, and that text was being thrown away. It is now written to
`ollama.log` beside the other logs and included in the Diagnose zip, along
with the "describe in English" scanner's status file, which the zip also
did not carry.

**The test suite now runs on Windows for every change**, not only Linux.

## v1.8.20 — 22 Sep 2026

Two more reports from the Windows installation.

**Your "Describe in English" sentences now stay on screen after you save
them.** They were being saved, and the engine was scanning for them, but the
camera list the Rules tab reads never carried them — so every sentence
vanished from the panel the moment it refreshed and looked as if it had not
saved. The list now includes them, and the "N English rules" count on each
camera card is right.

**Alert replays no longer show as missing when they exist.** Three different
situations all produced "No recorded evidence is available":

- A critical alert opened straight after it fired. Its record exists about
  20 seconds before its evidence does; the card now says the evidence is
  still being written and fills in by itself when it lands.
- A clip this machine cannot play. Instead of a silent black box, the card
  falls back to the recorded frames and says which format the clip was
  saved in.
- The card asked for the clip by its evidence folder, which a just-pushed
  alert does not have yet. It now asks by the alert's own id.

## v1.8.19 — 22 Sep 2026

**Loitering, intrusion and restricted-area alerts need a zone — and Argus
now says so.** These alerts only exist inside a zone you have drawn on the
camera. A new site has none, so those alerts could not fire, and nothing on
screen explained why. Each camera's Detectors and Rules tabs now tell you
when it has no zone, and setup lists every camera still without one.

**Watch the whole view in one click.** If you want loitering on a camera
without drawing anything, open its Zones tab and choose "Watch the whole
view". Argus covers the camera's full picture as a zone with the dwell time
you set. You can still draw a precise area instead, or later.

## v1.8.18 — 21 Sep 2026

Three things found on a Windows installation, and one on a gate camera.

**Replay videos now play in the app on Windows.** Evidence clips were being
written in a format Windows machines could encode but the app's own player
could not decode, so the clip existed, reached your phone, and showed as a
blank box in the app. Clips are now converted to standard H.264 as they are
written, on every platform.

**Finish setup is no longer locked by the readiness checks.** The button
stayed grey until every check had passed, which on a fresh installation
meant until the 3.3 GB AI model had finished downloading — with nothing on
screen saying so. You can now finish with a camera connected; anything still
needing attention is listed instead of blocking you.

**"Describe in English" rules explain themselves when the AI is not ready.**
Your sentences were always saved. But until the on-device model is
downloaded they cannot be evaluated, and the panel showed a raw connection
error that read as if the rules were broken. It now says the model is not
installed yet, where to download it, and that your rules are safe.

**One vehicle through a gate is one alert.** A car waiting at a barrier arm
could be counted several times as the tracker lost and re-found it. A vehicle
re-appearing where one just crossed, going the same way, is recognised as the
same vehicle. On the test footage this took one car from nine alerts to one.

## v1.8.17 — 21 Sep 2026

**Set up phone alerts from the app.** Where alerts go used to be a single text
box you had to already know how to fill in: Telegram meant typing a
colon-separated string that was documented nowhere, and typing just
`telegram` quietly did nothing at all. Now Settings → Alerts is a list of
channels. Tick Telegram and the app walks you through getting a bot token
from BotFather and your chat ID; tick WhatsApp and it walks you through
Twilio. Tokens are masked as you type, a group chat ID is recognised, and
**Send test alert** confirms it arrives before you rely on it. WhatsApp could
not previously be set up on an installed machine at all.

**Alerts are a setup step.** First-run setup now asks where alerts should go,
and asks you to name your locations before it asks which location a camera
belongs to — so nothing is requested before it exists.

**Argus now says when it has delivered an alert.** Delivery to a phone used to
leave no record on success, so a log could not tell a delivered alert from
one that was lost. Each delivery is now recorded with the alert and the
destination, and a failure names the alert it lost.

**Housekeeping you should know about.** A test now prevents any credential
being committed inside a configuration file. If you had put a bot token into
a config by hand because there was no better place, move it to Settings →
Alerts, which stores it in your own data directory rather than a file that
travels with the software.

## v1.8.16 — 20 Sep 2026

**The on-device AI downloads again.** Argus verifies alerts with an AI model
that runs entirely on your machine. It is about 3.3 GB and is fetched once,
because shipping it inside the installer would double the download for people
who already have it. The app used to start that download as soon as you began
setup, so it finished while you did everything else. Since the interface was
rebuilt, it never started — and nothing on screen said so. An installation
would detect people and vehicles correctly but leave every alert marked
unverified, and scene mapping, which uses the same model, would appear not to
work at all.

The download now starts when setup opens and shows its progress while you
carry on. The Verification step has a Download button if you would rather
start it yourself, and an interrupted download resumes where it stopped.

**If you installed v1.8.15 or earlier and never saw a 3.3 GB download, this
is why.** Upgrade, open setup, and let it run once. Your cameras, accounts
and recorded events are untouched.

## v1.8.15 — 20 Sep 2026

**Your cameras show a picture again.** Adding a camera and pressing Start
Monitoring stopped the detection engine dead, on every camera, not only the
new one. The camera would test as reachable, the button would quietly flip
back, and no video ever arrived. The cause: a camera added through the app
carries no detection settings until you pick what the site is for, and the
engine treated that as a fatal error rather than a choice you had not made
yet. A camera with no use case selected now runs the general detection set,
and choosing a use case still replaces it. **Upgrade if you added a camera on
v1.8.14 and saw no picture — nothing was wrong with your camera.**

**When monitoring stops, Argus now tells you why.** The app reported a
crashed engine exactly the same way it reported one you had stopped on
purpose: stopped, no explanation. It now shows the reason and where the log
is, so a problem arrives as a message on screen instead of a black rectangle.

**Adding your first camera no longer asks where it goes before anywhere
exists.** On a new site the Branch and Area fields were required but empty,
because branches are created elsewhere. The first camera now starts unplaced,
and the app says where to place it once you have set your locations up.

## v1.8.14 — 18 Sep 2026

**Argus now opens on your site, not on a sample one.** Every launch started in
a demonstration workspace: a fixture site with invented cameras and invented
incidents, already signed in as nobody. A first-time user never saw the
sign-up screen, never saw sign-in, and never saw site setup — the only way to
the real thing was a small Demo / Local engine switch in the top bar that
nothing drew attention to. The desktop app now opens on your own installation,
which on a new machine means creating your owner account. The demonstration
workspace is still there, one click away, and it stays selected if you choose
it.

**Setting up a new site walks you through it.** Sign in to a site that has not
been set up yet and Argus opens the setup wizard — cameras, scenes and zones,
what the site is for, which detectors to run, how verification behaves, and a
final check — instead of an empty dashboard with no obvious first move.

**Connecting a camera no longer requires knowing RTSP.** Adding a camera asked
for a stream address, which is fine if you already know that a Hikvision
substream lives at /Streaming/Channels/102 and a Tapo at /stream2, and useless
if you don't. Enter the camera's IP address, pick the make, and type the
username and password: Argus builds the address. The password is masked while
you type it, with a reveal button, and passwords containing symbols such as @
or : are handled correctly instead of corrupting the address. You can still
type a full address by hand.

**Camera discovery on the Add Camera screen works again.** Scanning the network
for cameras had been returning "no such camera" — the request was being
answered by the wrong handler. Anyone who tried it got an error with no
explanation.

**Everything the engine can do is reachable from the interface again.** Moving
to the new interface left a long tail of capabilities with no way to invoke
them: PDF incident reports, evidence export, legal holds, weekly summaries,
shift handovers, backups, the audit trail, account administration, model
downloads and per-camera rules. All of it is available from the System screen.
Several of these deserve screens of their own and will get them; being
reachable comes first.

## v1.8.13 — 17 Sep 2026

**The new interface is finally the one you install.** Argus has had a rebuilt
desktop UI since 9 September, and every release since shipped the old one
anyway: the frontend was merged as source but no build step ever compiled or
packaged it, and nothing in CI checked. Five releases went out carrying an
interface nobody had been demonstrating. The installer now *is* that new UI,
with the detection engine packaged inside it.

**What changes when you upgrade.** Your site configuration, accounts and
recorded events stay where they are and open as before — the new shell reads
the same per-user data directory the old one wrote to. The Windows installer
keeps its name and installs over the top. The old interface is still inside
the installation as a support fallback; nothing launches it.

**Account recovery now works on an installed machine.** Recovery had only ever
existed as a script for someone with the source code and a Python environment,
which no customer has — a locked-out installation had no way back in. The
shipped engine exposes the same recovery flow, with the same protections, and
the app shows the exact command to run.

**A build can no longer ship without its interface.** The release now fails if
the packaged app is missing the renderer, either engine binary, or their model
weights — the check whose absence caused all of this.

## v1.2.0 — 28 Aug 2026

**Windows customers get a real installer.** `argus-windows-setup.exe` now sits
next to the portable zip: Start Menu entry, desktop shortcut, an uninstaller,
and new versions install over old ones instead of beside them. SmartScreen
still warns once — that is the missing signing certificate, not the packaging.

**Live streams no longer fast-forward.** Segmented sources (HLS and similar)
deliver video in multi-second bursts; the wall used to freeze for a segment and
then replay it at decode speed. Bursts now play out smoothly at content rate a
few seconds behind live, catching up quietly when genuinely behind, with lag
hard-capped at eight seconds. Measured on the same public feeds: 0.7-1.2 fps
with five-second lurches before, ~11 fps steady after. Real RTSP cameras
deliver continuously and ride the same path at near-zero lag.

**Several plain-English rules on one camera now all fire.** The scanner used to
ask the model for THE threat — singular — so whichever rule it found most
salient answered every cycle and the rest never fired at all. Every rule is now
checked independently, each with its own cooldown. Also, a threat the model
invents no longer slips through by sharing one generic word with a real rule.

## v1.1.1 — 27 Aug 2026

**Update if you are on v1.1.0.** One user-visible fix, and it is the kind that
looks like a dead product.

- **The live wall no longer goes black after the engine restarts.** Every engine
  restart — switching feeds, a watchdog respawn, or the first start racing the
  first render — issues a new frame-publisher port and token. The wall's tiles
  kept requesting the old one and stayed blank, showing a broken-image icon,
  while clicking a tile to zoom it worked normally. The console's own self-heal
  had already spotted the failure and fetched the new port, but never repainted
  the tiles with it. Tiles that are stale or dead are now repointed within about
  two seconds; healthy tiles are left connected, so one dead camera cannot
  restart every stream on the wall.

Also in this release, affecting measurement rather than the running product:

- **A measurement run now exercises the detector it is named after.** Selecting
  a threat with `--kind` chose which clips to load but not which detectors to
  run, so a "weapons" run could complete, report clean figures with confidence
  intervals, and be describing the shoplifting detector's opinion of the
  footage. The gate was also told every clip was a retail shop, including
  street footage, which points it away from what is being measured. Both are
  fixed and tested. Numbers published before this release were produced under
  the old scene context and are being re-measured.

## v1.1.0 — 27 Aug 2026

**The live wall works.** v1.0.0 shipped with the frame publisher answering
HTTP/1.0, and no browser will progressively render a multipart stream over
HTTP/1.0 — so every tile was blank in that build.

- The wall streams MJPEG instead of polling eleven times a second
- One capture path for every platform; Windows was buffering itself into lag
- Live-edge draining, so a burst from a segmented stream no longer becomes
  permanent delay
- The alerts list ships ~70 KB per tab switch instead of 189 MB
- The 3.3 GB verification model downloads *while* you complete setup, not after
- Several plain-English rules per camera, applied without a restart
- Retention actually deletes expired evidence. It never once did: evidence paths
  were stored relative to the working directory and resolved against the output
  root, so the safety check refused every deletion and the disk simply filled.
  The retention period shown in the app was not being enforced.
- Accounts and the audit log stay global across feeds — per-feed stores had
  fragmented them, so nobody could sign in from a phone on a non-default feed

Installers are smaller across the board: Windows 511 MB, macOS 636 MB,
Linux 859 MB.

## v1.0.0 — 25 Aug 2026

First packaged release: one-click installers for macOS, Windows and Linux, with
the detection engine, the local verification model runtime, and the operator
console in a single download.

---

### Known limitations, all versions

- **The installers are not code-signed.** macOS and Windows will warn that the
  developer is unidentified. Signing certificates are not yet purchased.
- **Segmented live sources (HLS) look jerky.** Argus stays at the live edge, so
  on a source that delivers five seconds of video at a time there is nothing new
  to show between segments. Measured on public YouTube camera feeds: ~1 fps with
  5-second holds, against 11–12 fps on local video and on a continuous stream
  through the identical pipeline. Real RTSP cameras deliver continuously and do
  not behave this way.
- **Seven detectors are built and demonstrable but not yet validated** — panic
  running, person collapsed, weapons, violence, camera tampering, loitering, and
  custom plain-English rules. See `docs/NUMBERS.md`, which states what has been
  measured and what has not.
