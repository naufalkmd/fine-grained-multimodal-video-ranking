"""Deterministic counterfactual-caption *diagnostics*, never verified labels.

Only supplied captions are transformed. No external text, language model, visual
judge, or teacher prediction is consulted. Run this on the user's free Colab T4
alongside evaluation; the module itself needs only Python's standard library.

The original positive stays byte-for-byte unchanged. Minimal edits change one
span: action, state, relation, color, shape, or object noun. The old generic
show/add verb-transfer path is disabled. Genuine event appends are produced by
the separate Qwen generator with official-action-supported positive extensions.
No caption/metadata absence proves visual falsity.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import random
import re

VERSION = "scoped-attributes-and-minimal-v2"

# Inflection is explicit: a bare substring replacement could corrupt nouns,
# negation scope, or a different verb. These are transformation rules, not data.
ACTION_FAMILIES = {
    "open_close": (("open", "close"), ("opens", "closes"), ("opening", "closing")),
    "lift_lower": (("lift", "lower"), ("lifts", "lowers"), ("lifting", "lowering")),
    "push_pull": (("push", "pull"), ("pushes", "pulls"), ("pushing", "pulling")),
    "button_unbutton": (("button", "unbutton"), ("buttons", "unbuttons"), ("buttoning", "unbuttoning")),
    "fasten_unfasten": (("fasten", "unfasten"), ("fastens", "unfastens"), ("fastening", "unfastening")),
}
STATE_PAIRS = (("open", "closed"), ("empty", "full"))
UNSAFE = re.compile(r"\b(?:not|never|no|without|pretend\w*|mim\w*|try|tries|trying|attempt\w*|almost|apparently|possibly)\b", re.I)
ACTOR = r"(?:they|he|she|a person|the person|a hand|the hand|the man|the woman)"
ADVERB = r"(?:(?:then|next|now|slowly|carefully|briefly)\s+)?"
ACTION_CONTEXT = re.compile(r"(?:\b" + ACTOR + r"\s+(?:(?:is|are)\s+)?" + ADVERB + r"|\band\s+" + ADVERB + r")$", re.I)
DIRECT_OBJECT = re.compile(r"^\s+(?:up\s+)?(?:the|a|an|this|that|their|his|her|its|it|them|both|two|one|some|each|several|all|object_[a-z]+)\b", re.I)
STATE_CONTEXT = re.compile(r"\b(?:is|are|remains?|stays?)\s+(?:completely\s+)?$", re.I)
SPATIAL = re.compile(r"\b(?:to|on|at)\s+the\s+(left|right)\s+of\b", re.I)
WORD = re.compile(r"[A-Za-z]+(?:_[A-Za-z]+)?")
REFERENTIAL = re.compile(r"\b(?:this|that|these|those|another|again|same|former|latter|previous|before|after|while|until|once)\b", re.I)
SIMPLE_EVENT = re.compile(r"^(?P<marker>(?:(?:Finally|Next),\s+)?)(?P<actor>They|The person|A person|The hand|A hand)\s+(?P<adverb>(?:(?:then|next)\s+)?)(?P<verb>[a-z]+)\s+", re.I)
STOP = set("a an the they he she person man woman hand hands his her their to of in on at from with and or as by for up down out into over under near beside front back side left right small large little thin thick long short red blue green yellow white black brown gray grey pink orange purple dark light bright pale metal wooden plastic paper table surface tabletop camera view visible object item holds hold holding places place placing sets set brings bring moves move moving lifts lift lifting lowers lower lowering opens open opening closes close closing pushes push pulling pull pulls pressing press presses turns turn turning takes take taking puts put putting raises raise raising keeps keep keeping picks pick picking releases release releasing rests rest resting returns return returning displays display displaying removes remove removing lays lay laying tips tip tipping tilts tilt tilting shows show showing begins begin beginning starts start starting ends end ending repeatedly gently slowly carefully briefly upright flat horizontal vertical toward away through".split())


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def _words(text):
    return [w.lower() for w in WORD.findall(text)]


def _sentences(text):
    """Offsets into original text, preserving all bytes when replacing a span."""
    spans = []
    start = 0
    for match in re.finditer(r"[.!?](?:\s+|$)", text):
        end = match.start() + 1
        if text[start:end].strip():
            a = start + len(text[start:end]) - len(text[start:end].lstrip())
            spans.append((a, end, text[a:end]))
        start = match.end()
    if text[start:].strip():
        a = start + len(text[start:]) - len(text[start:].lstrip())
        spans.append((a, len(text), text[a:]))
    return spans


def _case(word, source):
    return word.upper() if source.isupper() else word.capitalize() if source.istitle() else word


def _tokens(tokenizer, text):
    value = tokenizer(text, add_special_tokens=False)
    ids = value["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        raise ValueError("Tokenizer must return one unbatched input ID list")
    return list(ids)


def _content(text):
    # Lexical compatibility filter, not an object detector or truth verifier.
    return {w for w in _words(text) if w not in STOP and len(w) > 2}


def _object_head(text):
    """Conservative determiner-led direct object head, from supplied wording.

    The lexical head is a compatibility cue only. It cannot identify an actual
    object, infer affordances perfectly, or establish that an action is absent.
    """
    match = SIMPLE_EVENT.match(text)
    if not match:
        return None
    body = text[match.end():]
    noun_phrase = re.match(r"(?:a|an|the)\s+(.+)", body, re.I)
    if not noun_phrase:
        return None
    phrase = re.split(r"\b(?:in|on|at|to|from|with|by|of|beside|near|over|under|toward|into|and|then|up|down|out|off|back|while|before|after)\b|[,.;:]", noun_phrase.group(1), maxsplit=1, flags=re.I)[0]
    words = _words(phrase)
    if not 1 <= len(words) <= 5:
        return None
    head = words[-1]
    if head in {"edge", "side", "top", "bottom", "end", "surface", "front", "back", "position", "part", "pair", "pile", "stack", "bunch", "set", "piece"}:
        return None
    if head.endswith("s") and not head.endswith("ss") and len(head) > 4:
        head = head[:-1]
    return head


def _event(text):
    """Accept only a short, simple standalone actor-action sentence.

    The actor and leading temporal marker are matched between both continuations.
    Object pronouns must follow an explicit noun phrase inside the same sentence.
    This is a lexical dependency heuristic, never a coreference/truth guarantee.
    """
    match = SIMPLE_EVENT.match(text)
    if not match or not text.endswith(".") or UNSAFE.search(text) or REFERENTIAL.search(text):
        return None
    if any(mark in text for mark in (";", ":", "?", "!")) or re.search(r"\b(?:or|but)\b", text, re.I):
        return None
    words = _words(text)
    if not 5 <= len(words) <= 40:
        return None
    body = text[match.end():]
    for pronoun in re.finditer(r"\b(?:it|its|them|their)\b", body, re.I):
        # A determiner plus content-bearing word before the pronoun is required.
        previous = body[:pronoun.start()]
        if not re.search(r"\b(?:a|an|the)\s+[a-z]", previous, re.I) or not _content(previous):
            return None
    content = _content(text)
    if not content:
        return None
    signature = [match.group(name).lower().strip() for name in ("marker", "actor", "adverb")]
    return {"verb": match.group("verb").lower(), "content": content, "signature": signature,
            "verb_span": list(match.span("verb")), "object_head": _object_head(text)}


def _minimal(text, actions, rejects):
    tokens = set(_words(text))
    for family, pairs in ACTION_FAMILIES.items():
        left = {p[0] for p in pairs}
        right = {p[1] for p in pairs}
        # Completed forms count as conflicting evidence even though not edited.
        if family == "open_close":
            left |= {"opened"}; right |= {"closed"}
        if tokens & left and tokens & right:
            rejects["opposite_action_words_in_same_caption"] += 1
            continue
        if actions:
            label_words = set(_words(" ".join(actions)))
            if label_words & left and label_words & right:
                rejects["opposite_action_labels_in_same_episode"] += 1
                continue
        replacements = dict(pairs)
        replacements.update({b: a for a, b in pairs})
        for start, end, sentence in _sentences(text):
            if UNSAFE.search(sentence):
                rejects["uncertain_or_negated_sentence"] += 1
                continue
            for match in WORD.finditer(sentence):
                old = match.group().lower()
                if old not in replacements:
                    continue
                prefix = sentence[:match.start()]
                copula = re.search(r"\b(?:is|are)\s+" + ADVERB + r"$", prefix, re.I)
                # 'They are open' is a state, not the progressive 'are opening'.
                # The direct-object gate also avoids '... and buttons ...' nouns.
                if (not ACTION_CONTEXT.search(prefix)
                        or bool(copula) != old.endswith("ing")
                        or not DIRECT_OBJECT.match(sentence[match.end():])):
                    rejects["ambiguous_action_grammar"] += 1
                    continue
                new = _case(replacements[old], match.group())
                a, b = start + match.start(), start + match.end()
                yield {"type": "action_" + family, "start": a, "end": b,
                       "old": text[a:b], "new": new,
                       "negative": text[:a] + new + text[b:], "rule": "explicit_actor_verb_single_span"}
    for start, end, sentence in _sentences(text):
        if UNSAFE.search(sentence):
            continue
        for left, right in STATE_PAIRS:
            if left in tokens and right in tokens:
                rejects["opposite_states_in_same_caption"] += 1
                continue
            mapping = {left: right, right: left}
            for match in WORD.finditer(sentence):
                word = match.group().lower()
                if word not in mapping or not STATE_CONTEXT.search(sentence[:match.start()]):
                    continue
                a, b = start + match.start(), start + match.end()
                new = _case(mapping[word], match.group())
                yield {"type": "state_" + left + "_" + right, "start": a, "end": b,
                       "old": text[a:b], "new": new,
                       "negative": text[:a] + new + text[b:], "rule": "copular_state_single_span"}
        for match in SPATIAL.finditer(sentence):
            nearby = sentence[max(0, match.start() - 24):min(len(sentence), match.end() + 30)]
            if re.search(r"\b(?:hand|hands|arm|arms|leg|legs|foot|feet)\b", nearby, re.I):
                rejects["body_side_ambiguity"] += 1
                continue
            a, b = start + match.start(1), start + match.end(1)
            old = text[a:b]
            new = _case("right" if old.lower() == "left" else "left", old)
            yield {"type": "spatial_left_right", "start": a, "end": b,
                   "old": old, "new": new, "negative": text[:a] + new + text[b:],
                   "rule": "explicit_object_relation_single_span"}


COLORS = ('black', 'white', 'red', 'blue', 'green', 'yellow', 'orange', 'purple',
          'pink', 'brown', 'gray', 'grey')
SHAPE_NOUNS = ('triangle', 'square', 'circle', 'rectangle')
SHAPE_ADJECTIVES = ('triangular', 'square', 'circular', 'rectangular')
# Avoid near-synonym/hypernym pairs such as cup/mug and pen/marker.
OBJECT_GROUPS = (('cup', 'bowl'), ('pen', 'pencil'), ('bag', 'box'))
PLURAL = {word: word + 's' for group in OBJECT_GROUPS for word in group}
PLURAL.update({word: word + 's' for word in SHAPE_NOUNS})
PLURAL['box'] = 'boxes'
HEADS = set(PLURAL) | set(PLURAL.values()) | set('bottle bottles jar jars mug mugs marker markers paper sheet sheets card cards container containers lid lids cap caps book books cloth cloths towel towels shirt shirts fabric tote totes wallet wallets cable cables spoon spoons fork forks plate plates glass glasses tray trays tin tins ball balls block blocks case cases tube tubes packet packets cushion cushions sock socks shoe shoes glove gloves'.split())
MODIFIERS = set(COLORS) | set(SHAPE_ADJECTIVES) | set('small large little thin thick big tall short long dark light bright pale clear transparent opaque plastic metal metallic wooden glass paper cardboard fabric folded crumpled round flat striped printed checked empty full open closed black-capped'.split())
NP = re.compile(r'\b(?P<determiner>a|an|the|this|that|their|his|her|two|three|one|some)\s+'
                r'(?P<body>(?:(?:' + '|'.join(sorted(MODIFIERS, key=len, reverse=True)) + r')\s+){0,5}'
                r'(?P<head>' + '|'.join(sorted(HEADS, key=len, reverse=True)) + r'))\b', re.I)


def _attribute_aliases(word):
    for noun, adjective in zip(SHAPE_NOUNS, SHAPE_ADJECTIVES):
        family = {noun, PLURAL[noun], adjective}
        if word in family:
            return family
    for noun, plural in PLURAL.items():
        if word in {noun, plural}:
            return {noun, plural}
    return {'gray', 'grey'} if word in {'gray', 'grey'} else {word}


def _attribute_candidates(text, vocabulary, rejects):
    """One scoped noun-phrase edit; no global replacement or noun/verb guessing.

    Grammar and lexical filtering only: target attribute/object could still be
    present in the images. Replacement vocabulary must occur in supplied TRAIN.
    """
    all_words = set(_words(text))
    for start, _, sentence in _sentences(text):
        if UNSAFE.search(sentence):
            continue
        for phrase in NP.finditer(sentence):
            head = phrase.group('head').lower()
            body_start = phrase.start('body')
            mods = list(WORD.finditer(sentence, body_start, phrase.start('head')))
            edits = []
            # Multiple colors in the same phrase can describe a pattern. Skip.
            color_words = [m for m in mods if m.group().lower() in COLORS]
            if len(color_words) == 1:
                m = color_words[0]
                edits += [('attribute_color', m.start(), m.end(), new)
                          for new in COLORS
                          if {m.group().lower(), new} != {'gray', 'grey'}]
            elif len(color_words) > 1:
                rejects['multicolor_phrase'] += 1
            shape_words = [m for m in mods if m.group().lower() in SHAPE_ADJECTIVES]
            if len(shape_words) == 1:
                m = shape_words[0]
                edits += [('attribute_shape', m.start(), m.end(), new) for new in SHAPE_ADJECTIVES]
            if head in SHAPE_NOUNS or head in {PLURAL[x] for x in SHAPE_NOUNS}:
                plural = head not in SHAPE_NOUNS
                edits += [('attribute_shape', phrase.start('head'), phrase.end('head'),
                           PLURAL[new] if plural else new) for new in SHAPE_NOUNS]
            for group in OBJECT_GROUPS:
                forms = group if head in group else tuple(PLURAL[x] for x in group)
                if head in forms:
                    edits += [('object_noun', phrase.start('head'), phrase.end('head'), new) for new in forms]
            for kind, a, b, replacement in edits:
                old = sentence[a:b]
                if replacement == old.lower():
                    continue
                if _attribute_aliases(replacement) & all_words:
                    rejects['replacement_already_in_caption'] += 1
                    continue
                if not vocabulary[replacement]:
                    rejects['replacement_absent_from_train'] += 1
                    continue
                # Avoid silently changing articles or generating a/an errors.
                if a == body_start and phrase.group('determiner').lower() in {'a', 'an'}:
                    article = phrase.group('determiner').lower()
                    needed = 'an' if replacement[0] in 'aeiou' else 'a'
                    if article != needed:
                        rejects['article_agreement_change_required'] += 1
                        continue
                replacement = _case(replacement, old)
                aa, bb = start+a, start+b
                yield {'type': kind, 'start': aa, 'end': bb, 'old': text[aa:bb], 'new': replacement,
                       'negative': text[:aa]+replacement+text[bb:], 'rule': 'single_scoped_noun_phrase_edit',
                       'object_scope_span': [start+phrase.start(), start+phrase.end()],
                       'object_scope_text': phrase.group(), 'source_object_head': head,
                       'limitations': 'Noun phrase grammar is screened; visual falsity is not verified.'}


def build_suite(manifest, split, tokenizer, action_labels=None, seed=2031,
                max_minimal_per_episode=4, max_additions_per_episode=0):
    """Generate minimal candidates only. Actual append generation is separate."""
    if max_minimal_per_episode < 0:
        raise ValueError('Candidate cap must be nonnegative')
    if max_additions_per_episode != 0:
        raise ValueError('Weak event replacement is disabled; use the genuine Qwen append generator')
    episodes = {e['episode_id']: e for e in manifest['episodes']}
    if len(episodes) != len(manifest['episodes']):
        raise ValueError('Duplicate manifest episode IDs')
    train, val = list(split['train_ids']), list(split['val_ids'])
    if len(set(train)) != len(train) or len(set(val)) != len(val) or set(train) & set(val):
        raise ValueError('Split must have disjoint unique train/validation episode IDs')
    if not set(train + val) <= episodes.keys():
        raise ValueError('Split references absent episodes')
    action_labels = action_labels or {}
    vocabulary = Counter(w for ep in train for w in _words(episodes[ep]['description']))
    rejects, records, per_episode = Counter(), [], []
    for ep in val:
        text = episodes[ep]['description']
        rng = random.Random(int(_digest([seed, ep])[:16],16))
        candidates = (list(_minimal(text,action_labels.get(ep,[]),rejects)) +
                      list(_attribute_candidates(text,vocabulary,rejects))) if max_minimal_per_episode else []
        # Round-robin by semantic category prevents plentiful color variants
        # from consuming every slot. No model scores are used in selection.
        buckets = defaultdict(list)
        for candidate in candidates:
            category = candidate['type'].split('_')[0] if candidate['type'].startswith(('action_', 'state_')) else candidate['type']
            buckets[category].append(candidate)
        keys = sorted(buckets)
        rng.shuffle(keys)
        for group in buckets.values(): rng.shuffle(group)
        ordered = []
        while any(buckets.values()):
            for key in keys:
                if buckets[key]: ordered.append(buckets[key].pop())
        selected, seen = [], {text.casefold()}
        p = _tokens(tokenizer,text)
        for candidate in ordered:
            negative = candidate['negative']
            if negative.casefold() in seen: continue
            seen.add(negative.casefold())
            n = _tokens(tokenizer,negative)
            if p == n:
                rejects['identical_full_token_inputs'] += 1
                continue
            common = next((i for i,(a,b) in enumerate(zip(p,n)) if a != b),min(len(p),len(n)))
            evidence = {k:v for k,v in candidate.items() if k != 'negative'}
            evidence.update(source_episode_id=ep,
                            replacement_word_train_positive_count=vocabulary[candidate['new'].lower()],
                            visual_truth_verified=False)
            selected.append({'id':'hard_'+_digest([VERSION,seed,ep,candidate['type'],negative])[:24],
                             'episode_id':ep,'family':'minimal','type':candidate['type'],
                             'positive_text':text,'negative_text':negative,'semantic_status':'rule_assumed',
                             'evidence':evidence,'positive_token_count':len(p),'negative_token_count':len(n),
                             'positive_sentence_count':len(_sentences(text)),
                             'negative_sentence_count':len(_sentences(negative)),
                             'base64_identical':p[:63]==n[:63],'first_changed_token_index':common,
                             'change_after_base_content_window':common>=63})
            if len(selected) >= max_minimal_per_episode: break
        records.extend(selected)
        per_episode.append({'episode_id':ep,'pairs':{'minimal':len(selected)},'no_eligible_candidate':not selected})
    return {'records':records,'audit':{
        'version':VERSION,'seed':seed,'parent_validation_episodes':len(val),'parent_validation_ids':val,
        'train_vocabulary_episodes':len(train),'max_minimal_per_episode':max_minimal_per_episode,
        'max_additions_per_episode':0,'records':len(records),'per_episode':per_episode,
        'coverage':{'minimal':{'pairs':len(records),'episodes':len({r['episode_id'] for r in records})}},
        'type_counts':dict(Counter(r['type'] for r in records)),'exclusion_counts':dict(rejects),
        'base64_identical_pairs':sum(r['base64_identical'] for r in records),
        'length_delta_histogram':{'minimal':dict(Counter(str(r['negative_token_count']-r['positive_token_count']) for r in records))},
        'zero_positive_frequency_replacements':dict(Counter(r['evidence']['new'].lower() for r in records
            if not r['evidence']['replacement_word_train_positive_count'])),
        'selection':'Seeded round-robin across semantic categories; no model-based filtering',
        'semantic_warning':'Rule-assumed counterfactuals, not visually verified truth. Different objects/colors may coexist.',
        'event_construction':'Old event replacement is disabled. Genuine literal appends are generated separately.',
        'data_policy':'Provided captions and train-only replacement vocabulary; no external captions or teacher labels.',
        'evaluation_warning':'Fixed development-video diagnostics, not an independent test or leaderboard estimate.'}}

