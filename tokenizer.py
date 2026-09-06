# ============================================================
# TINYGPT
# PHASE 19 — KNOWLEDGE-AWARE TOKENIZER V8
# ============================================================

import os
import re


# ============================================================
# 1. SPECIAL TOKENS
# ============================================================

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"

SPECIAL_TOKENS = [
    PAD_TOKEN,
    UNK_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN
]


# ============================================================
# 2. PUNCTUATION
# ============================================================

PUNCTUATION = [
    ".",
    ",",
    "?",
    "!",
    ":",
    ";",
    "'",
    '"',
    "(",
    ")",
    "-"
]


# ============================================================
# 3. TOKENIZATION
# ============================================================

def tokenize(text):

    return re.findall(
        r"[A-Za-z0-9]+|[.,!?;:'\"()-]",
        text
    )


# ============================================================
# 4. LOAD KNOWLEDGE
# ============================================================

KNOWLEDGE_FILE = "knowledge.txt"


def load_knowledge():

    if not os.path.exists(KNOWLEDGE_FILE):

        print()
        print("WARNING:")
        print("knowledge.txt was not found.")
        print(
            "Tokenizer will use only special tokens "
            "and punctuation."
        )

        return ""

    with open(
        KNOWLEDGE_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        return file.read()


# ============================================================
# 5. BUILD VOCABULARY
# ============================================================

def build_vocabulary(text):

    vocabulary = []

    # --------------------------------------------------------
    # Special tokens first.
    # --------------------------------------------------------

    vocabulary.extend(
        SPECIAL_TOKENS
    )

    # --------------------------------------------------------
    # RAG control words.
    #
    # These words are required by the RAG training format.
    # --------------------------------------------------------

    rag_words = [
    "Context",
    "Question",
    "Answer",
    "context",
    "question",
    "answer",

    # Common chatbot question words
    "What",
    "Who",
    "Why",
    "How",
    "When",
    "Where",

    # Lowercase versions
    "what",
    "who",
    "why",
    "how",
    "when",
    "where",

    # Common conversation words
    "Hi",
    "Hello",
    "hi",
    "hello"
]

    # --------------------------------------------------------
    # Collect words exactly as they appear in knowledge.txt.
    # --------------------------------------------------------

    words = []

    for token in tokenize(text):

        if token not in PUNCTUATION:

            if token not in words:

                words.append(token)

    # --------------------------------------------------------
    # Add RAG control words.
    # --------------------------------------------------------

    for token in rag_words:

        if token not in words:

            words.append(token)

    # --------------------------------------------------------
    # Add knowledge words.
    # --------------------------------------------------------

    vocabulary.extend(
        words
    )

    # --------------------------------------------------------
    # Add punctuation.
    # --------------------------------------------------------

    vocabulary.extend(
        PUNCTUATION
    )

    # --------------------------------------------------------
    # Remove duplicates while preserving order.
    # --------------------------------------------------------

    vocabulary = list(
        dict.fromkeys(
            vocabulary
        )
    )

    return vocabulary


# ============================================================
# 6. LOAD KNOWLEDGE AND CREATE VOCABULARY
# ============================================================

KNOWLEDGE_TEXT = load_knowledge()

VOCABULARY = build_vocabulary(
    KNOWLEDGE_TEXT
)


# ============================================================
# 7. TOKEN IDS
# ============================================================

TOKEN_TO_ID = {

    token: index

    for index, token in enumerate(
        VOCABULARY
    )

}


ID_TO_TOKEN = {

    index: token

    for token, index in TOKEN_TO_ID.items()

}


VOCAB_SIZE = len(
    VOCABULARY
)


# ============================================================
# 8. ENCODE
# ============================================================

def encode(text):

    tokens = tokenize(
        text
    )

    token_ids = [

        TOKEN_TO_ID[
            BOS_TOKEN
        ]

    ]

    for token in tokens:

        # ----------------------------------------------------
        # Exact spelling first.
        # ----------------------------------------------------

        if token in TOKEN_TO_ID:

            token_ids.append(
                TOKEN_TO_ID[
                    token
                ]
            )

        # ----------------------------------------------------
        # Try lowercase.
        # ----------------------------------------------------

        elif token.lower() in TOKEN_TO_ID:

            token_ids.append(
                TOKEN_TO_ID[
                    token.lower()
                ]
            )

        # ----------------------------------------------------
        # Unknown token.
        # ----------------------------------------------------

        else:

            token_ids.append(
                TOKEN_TO_ID[
                    UNK_TOKEN
                ]
            )

    # --------------------------------------------------------
    # End of sequence.
    # --------------------------------------------------------

    token_ids.append(

        TOKEN_TO_ID[
            EOS_TOKEN
        ]

    )

    return token_ids


# ============================================================
# 9. DECODE
# ============================================================

def decode(token_ids):

    tokens = []

    for token_id in token_ids:

        token = ID_TO_TOKEN.get(
            token_id,
            UNK_TOKEN
        )

        if token in SPECIAL_TOKENS:

            continue

        tokens.append(
            token
        )

    # --------------------------------------------------------
    # Reconstruct readable text.
    # --------------------------------------------------------

    text = ""

    punctuation = set(
        PUNCTUATION
    )

    for token in tokens:

        if not text:

            text = token

        elif token in punctuation:

            text += token

        else:

            text += " " + token

    return text


# ============================================================
# 10. UNKNOWN TOKEN ANALYSIS
# ============================================================

def find_unknown_tokens(text):

    tokens = tokenize(
        text
    )

    unknown = []

    for token in tokens:

        if token in TOKEN_TO_ID:

            continue

        if token.lower() in TOKEN_TO_ID:

            continue

        if token not in unknown:

            unknown.append(
                token
            )

    return unknown


# ============================================================
# 11. TEST
# ============================================================

if __name__ == "__main__":

    print()

    print("======================================")
    print("       TINYGPT TOKENIZER V8")
    print("======================================")

    print()

    print("Knowledge file:")
    print(KNOWLEDGE_FILE)

    print()

    print("Vocabulary size:")
    print(VOCAB_SIZE)

    print()

    print("Special tokens:")

    print(
        f"PAD: {PAD_TOKEN} -> "
        f"{TOKEN_TO_ID[PAD_TOKEN]}"
    )

    print(
        f"UNK: {UNK_TOKEN} -> "
        f"{TOKEN_TO_ID[UNK_TOKEN]}"
    )

    print(
        f"BOS: {BOS_TOKEN} -> "
        f"{TOKEN_TO_ID[BOS_TOKEN]}"
    )

    print(
        f"EOS: {EOS_TOKEN} -> "
        f"{TOKEN_TO_ID[EOS_TOKEN]}"
    )


    # ========================================================
    # TEST 1
    # ========================================================

    test_text = "What is Pinocchio?"

    tokens = tokenize(
        test_text
    )

    encoded = encode(
        test_text
    )

    decoded = decode(
        encoded
    )

    print()

    print("======================================")
    print("TEST 1")
    print("======================================")

    print()

    print("Original:")
    print(test_text)

    print()

    print("Tokens:")
    print(tokens)

    print()

    print("Encoded:")
    print(encoded)

    print()

    print("Decoded:")
    print(decoded)


    # ========================================================
    # TEST 2
    # ========================================================

    test_text_2 = (
        "Pinocchio is a wooden puppet. "
        "Geppetto is a woodcarver."
    )

    encoded_2 = encode(
        test_text_2
    )

    decoded_2 = decode(
        encoded_2
    )

    print()

    print("======================================")
    print("TEST 2")
    print("======================================")

    print()

    print("Original:")
    print(test_text_2)

    print()

    print("Decoded:")
    print(decoded_2)


    # ========================================================
    # KNOWLEDGE TEST
    # ========================================================

    print()

    print("======================================")
    print("       KNOWLEDGE TEST")
    print("======================================")

    knowledge_tokens = tokenize(
        KNOWLEDGE_TEXT
    )

    unknown_tokens = find_unknown_tokens(
        KNOWLEDGE_TEXT
    )

    print()

    print("Knowledge tokens:")
    print(
        len(knowledge_tokens)
    )

    print()

    print("Unknown tokens:")
    print(
        len(unknown_tokens)
    )

    if unknown_tokens:

        print()

        print("Unknown vocabulary:")

        print(
            unknown_tokens
        )

    else:

        print()

        print(
            "All knowledge words are recognized "
            "by the tokenizer!"
        )


    # ========================================================
    # COMPLETE
    # ========================================================

    print()

    print("======================================")
    print("             COMPLETE")
    print("======================================")

    print()
