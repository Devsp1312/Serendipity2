# Serendipity

> A privacy-first wearable badge system that fosters organic, low-pressure social connections through ambient intelligence.

Serendipity passively learns who you are through ambient audio, builds an anonymous interest profile, and suggests real-world activities that naturally put like-minded people in the same place at the same time — without either person ever knowing they were matched.

---

## How It Works

The badge records ambient audio throughout the day. Overnight, while charging, the device processes everything locally on the Raspberry Pi 5: noise is removed, audio is transcribed, interest topics are extracted, and embeddings are generated. By morning, a fresh QR code appears on the e-ink display. The user scans it, opens the iOS app, and receives 4–5 personalized task suggestions for the day.

The serendipity happens in the real world. Two people independently receive the same suggestion, end up in the same place, and may — or may not — connect. No profiles are shown. No matches are revealed. No pressure.

---

## System Architecture

```
Ambient Audio
     │
     ▼
[Noise Removal]
     │
     ▼
[Transcription] ── faster-whisper
     │
     ▼
[Topic Extraction] ── Ollama (gemma2:2b)
     │
     ▼
[Embedding Generation] ── sentence-transformers
     │
     ▼
[Interest Profile JSON] ── stored on-device, audio deleted after 2 days
     │
     ▼
[QR Code on E-ink Display]
     │
     ▼
[iOS App Sync] ── anonymous token, no email required
     │
     ▼
[FastAPI Backend] ── cosine similarity matching
     │
     ▼
[Daily Task Suggestions] ── delivered to iOS app
```

---

## Hardware

| Component | Details |
|---|---|
| Compute | Raspberry Pi 5 (16GB RAM), RP3A0 module |
| Display | E-ink screen — shows QR code and mute indicator |
| Audio | Onboard microphone for ambient capture |
| Antenna | Meander-line antenna |
| Power | LiPo battery for all-day wear |
| Enclosure | Translucent side window |
| Controls | Physical mute button |

The badge is designed for all-day wear. During the day it only records audio — low power draw. All heavy compute runs overnight while plugged in.

---

## Privacy Design

Privacy is not a feature — it is the architecture.

- **Anonymous by default** — no email, no name, no account. The app generates a random token on first launch. If you lose your device, you start fresh.
- **On-device processing** — transcription, topic extraction, and embedding generation all run locally on the Pi 5. Nothing leaves the device except anonymized embeddings.
- **Audio deletion** — raw audio is automatically deleted after two days. The system never retains source recordings.
- **Mute button** — a physical button pauses recording at any time. The e-ink display shows a muted mic icon so anyone nearby can see the device is not recording.
- **No user profiles shown** — the app never displays other users, their interests, or their location. Users only see their own task suggestions.

---

## Software Stack

### On-Device (Raspberry Pi 5)
- `record.py` — ambient audio capture with privacy-preserving delete loop
- `transcribe.py` — audio transcription via faster-whisper
- `extract.py` — interest topic extraction via Ollama (gemma2:2b)
- Sentence-transformer embeddings stored as flat JSON

### Backend
- **FastAPI** — REST API for profile sync and cosine similarity matching
- **Flat JSON profiles** — lightweight, no database dependency

### iOS App (SwiftUI, iOS 17+)
- Apple Health / Wallet-style design using semantic SwiftUI colors
- Displays daily task suggestions
- Anonymous token authentication via QR code scan
- No user discovery or social graph

---

## Processing Pipeline Performance

| Step | Time |
|---|---|
| Noise removal | Near-instant |
| Transcription (1 hr audio) | ~5 minutes |
| Full 15-hr day | < 7 hours |
| Sleep window needed | 7+ hours |

Processing is intentionally throttled to avoid thermal issues on the Pi 5 form factor. If a user sleeps fewer than 7 hours, the previous day's profile carries over and the badge continues recording the new day.

---

## Recommendation System

The backend clusters users by cosine similarity on their interest embeddings. When a cluster is identified, the system generates a task suggestion — a real-world activity that would naturally attract that group.

### Example suggestions
- "Try rock climbing this weekend — Sunday mornings have the best vibe at the gym"
- "Check out the farmers market this Sunday around 10am"
- "Wear something yellow today — others will too"
- "Open mic night at the downtown pub on Thursday"
- "Leave your headphones off during your commute today"

### Third-party platform layer
Businesses and event organizers can plug into the recommendation layer under two strict rules:
- One notification per user per day maximum
- Permanent ban if more than 10% of recipients flag a recommendation as irrelevant

---

## Team

| Name | Role |
|---|---|
| Dev Patel | Systems, iOS app, ML pipeline |
| Emily DuPont | Backend, cloud infrastructure |
| Shivam Patel | Hardware, thermal management |

**Advisor:** Minning Zhu  
**Institution:** Rutgers University — ECE Capstone, Spring 2025

---

## Roadmap

- [x] Audio capture pipeline with privacy-preserving delete loop
- [x] On-device transcription and topic extraction
- [x] Sentence-transformer embedding generation
- [x] FastAPI backend with cosine similarity matching
- [x] SwiftUI iOS companion app (iOS 17+)
- [x] E-ink QR display and mute button
- [ ] Stage 2: On-device GPS tagging for location-aware recommendations
- [ ] Stage 2: Proximity-based opt-out signal between badges
- [ ] Multi-campus support

---

## Why Serendipity?

Every other social app makes the algorithm visible. You see your matches, you see who's going, you know why you're there. That creates pressure before you even walk in the door.

Serendipity works because you never know. You go rock climbing because you wanted to try it. The badge captured your real life — not your curated profile — and quietly put you in the right place at the right time.

The best friendships you've ever made weren't because an app told you to meet someone. They happened because you were both just there.

---

## License

MIT License — see `LICENSE` for details.
