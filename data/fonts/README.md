Place user-provided `.ttf`, `.otf`, `.ttc`, or `.otc` font files here.
Copy `font_use.example.txt` to `font_use.txt`, then describe when the VLM
should use each installed font.

The contents of `font_use.txt` are provided to the VLM as-is, so natural
language descriptions are fine. Font files and `font_use.txt` are ignored by
Git and are not shipped with tetolate.

My personal setup:

wild_words.ttf: talking, narrator, general text, fallback
wild_words_italic.ttf: thinking, monologue
roboto.ttf: computer text
architects_daughter.ttf: hand written text
orange_fizz_italic.ttf: sfx, loud shout

Obtain and license fonts separately. If no user fonts are installed, tetolate uses
the configured DejaVu Sans backup font.
