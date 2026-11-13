# Chat UI Spec (Nodi Assistant)

Visual reference: screenshots in `docs/design/` (Main light, Main dark, Mobile, Debug). Where this file and a screenshot disagree, ask me.

**When to build this:** milestone M4 (Chat). Plain HTML + CSS + vanilla JS served by FastAPI. No framework unless I approve one in `docs/decisions.md`. The Debug panel is a stretch goal for after M3, not part of the first build.

**All text in the screenshots is sample content.** Never hardcode answers, file names, scores or "12 documents". Everything comes from the API.

## 1. Layout (desktop, 1360 x 860 reference)

Three columns in a flex row that wraps on narrow screens:

| Column | Width | Content |
|---|---|---|
| Sidebar | 260px fixed | Brand, "New chat" button, recent chats, "documents indexed" card |
| Chat | flexible, content max 820px, centered | Header, messages, composer |
| Sources panel | 340px fixed | "Sources for this answer" cards |

Under about 1100px the sources panel drops below the chat. On phones (about 390px) the sidebar becomes a menu button and sources show as horizontally scrolling chips under the answer.

## 2. Design tokens

Fonts: Inter (body, 15px, line-height 1.5), Space Grotesk (headings 500/600), `ui-monospace` 12.5px for file names.

| Token | Light | Dark |
|---|---|---|
| bg (page) | #F6F7F5 | #0E1514 |
| surface | #FFFFFF | #162120 |
| ink (text) | #14211F | #E8EFED |
| body2 (secondary text) | #3F4F4C | #C5D3D0 |
| muted | #5B6B68 | #9DB0AC |
| line | #DDE3E0 | #2A3836 |
| line2 (control borders) | #CBD5D1 | #35504C |
| input border | #B7C4C0 | #3D5A55 |
| accent | #0F766E | #2DD4BF |
| on-accent (text/icon on accent) | #FFFFFF | #06201C |
| chip bg / border / text | #E3F1EE / #9CC9C2 / #0B5A53 | #12302C / #2B5F58 / #7EE8D8 |
| user bubble bg / text | #14211F / #FFFFFF | #24413D / #F2F7F6 |
| ok pill bg / text | #DDF1E8 / #0B4F34 | #12302C / #7EE8D8 |
| warning (superseded) border / bg / pill / text | #E3C48A / #FFFBF2 / #FBF0DC / #6B3F00 | #6B5320 / #1C1A12 / #2A2110 / #F3D9A0 |

Implement these as CSS custom properties on `:root` and `:root[data-theme="dark"]`.

## 3. Components

1. **Sidebar:** brand mark (32px rounded square, accent background, two-wave icon) + "Nodi Assistant"; "New chat" outlined button; "Recent" list (active item uses chip bg); bottom card showing the document count.
2. **Chat header:** conversation title, a small pill "Hybrid search + rerank", and the theme switch (see 5).
3. **User message:** right-aligned bubble, max 70% width, radius 16/16/4/16.
4. **Assistant message:** 32px round avatar, answer text (15px, line-height 1.65) with inline citation chips `[1]` `[2]`, a "Sources" row of file chips under it, then thumbs up / thumbs down buttons.
5. **"I don't know" state:** same avatar, but a dashed-border box with a bold "I don't know." and one muted line suggesting topics. It must look different from a normal answer.
6. **Composer:** a rounded card (radius 20) holding a 2-row textarea (16px), and below it a toolbar: scope button "All documents" on the left, hint "Enter to send, Shift + Enter for a new line" and a 48px round send button (accent, up-arrow icon) on the right. Helper line under the card.
7. **Sources panel:** one card per cited source: number badge, file name (mono), status pill, short snippet. Status "Current policy" (green) or "Superseded, not used" (amber). Footer note: when two sources disagree, the current one wins.
8. **Theme switch:** two icon-only round 44px buttons (sun, moon) in a pill group; selected one has accent background. No text. Each has `aria-label` and `title`.

## 4. Behaviour

- Enter sends; Shift+Enter inserts a newline; empty input disables send.
- Answer streams token by token (SSE). Show the text as it arrives; add citation chips when the final event arrives.
- Clicking `[n]` scrolls to and briefly highlights that source card.
- Chat history sent to the backend is the last 6 turns (see CLAUDE.md).
- Thumbs buttons POST feedback (stretch goal). The scope button is a metadata filter (stretch goal): render it disabled with a tooltip until the backend supports it.
- Theme: default from `prefers-color-scheme`, user choice saved in `localStorage` key `theme`.

## 5. API shape the UI expects (propose changes if needed)

```
POST /chat   { "message": str, "history": [{role, content}], "scope": "all" | "hr" | ... }
-> text/event-stream
   event: delta   data: {"text": "..."}
   event: done    data: {"sources": [{"id": 1, "file": str, "status": "current" | "superseded", "snippet": str}],
                         "answered": true | false}
```

`answered: false` renders the "I don't know" state. Decide `status` from document metadata, not by string-matching the file name.

## 6. Accessibility

- Touch targets at least 44px. Icon-only buttons need `aria-label`.
- Toggle buttons use `aria-pressed`. The textarea has a real `<label>`.
- Text contrast at least 4.5:1 in both themes. Status is always shown with a text label, never colour alone.
