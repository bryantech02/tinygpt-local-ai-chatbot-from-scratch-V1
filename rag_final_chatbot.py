# ============================================================
# TINYGPT
# FINAL FACTUAL RAG CHATBOT BY BRYAN TORCULAS
# ============================================================

import re
import torch
import torch.nn.functional as F

from sentence_transformers import SentenceTransformer
from tinygpt_transformer import TinyGPT
from tokenizer import VOCAB_SIZE


# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = torch.device("cpu")

KNOWLEDGE_FILE = "final_rag_embeddings.pt"
MODEL_FILE = "tinygpt_final_rag.pt"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

TOP_K = 5

# Minimum semantic similarity required after validation.
SIMILARITY_THRESHOLD = 0.40

# Additional margin required when the top result is competing
# with another record from the same topic.
MIN_SCORE_MARGIN = 0.015

MAX_SEQUENCE_LENGTH = 256


# ============================================================
# REFUSAL
# ============================================================

REFUSAL = (
    "I don't have enough information in my knowledge base "
    "to answer that accurately."
)


# ============================================================
# HEADER
# ============================================================

print()
print("======================================")
print("      TINYGPT FINAL RAG CHATBOT")
print("======================================")
print()

print("Device:")
print(DEVICE)


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print()
print("Loading embedding model...")

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL,
    device="cpu"
)

print("Embedding model loaded!")


# ============================================================
# LOAD KNOWLEDGE DATABASE
# ============================================================

print()
print("Loading final knowledge database...")

database = torch.load(
    KNOWLEDGE_FILE,
    map_location="cpu",
    weights_only=False
)

knowledge = database["knowledge"]

knowledge_embeddings = database["embeddings"]

knowledge_embeddings = F.normalize(
    knowledge_embeddings,
    p=2,
    dim=1
)

print("Knowledge database loaded!")
print()
print("Knowledge records:")
print(len(knowledge))


# ============================================================
# LOAD TINYGPT
# ============================================================

print()
print("Loading trained TinyGPT...")

checkpoint = torch.load(
    MODEL_FILE,
    map_location=DEVICE,
    weights_only=False
)

model = TinyGPT(
    vocab_size=VOCAB_SIZE,
    embedding_size=64,
    max_sequence_length=MAX_SEQUENCE_LENGTH,
    number_of_heads=4,
    number_of_layers=3,
    feed_forward_size=256,
    dropout=0.0
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(DEVICE)
model.eval()

print("TinyGPT loaded!")


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):
    text = text.lower().strip()

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# ROBUST QUERY NORMALIZATION
# ============================================================
#
# This layer is intentionally separate from normalize_text().
#
# normalize_text() performs basic cleaning.
#
# robust_query_normalize() additionally handles:
# - possessive forms such as "Japan's"
# - common conversational filler
# - repeated whitespace
# - punctuation/noise
#
# IMPORTANT:
# We preserve meaningful intent words such as:
# how, why, meaning, definition, used, use, capital, etc.
#
# The goal is NOT to rewrite the user's question into a
# hardcoded answer. The goal is to make imperfect English
# easier for the existing retrieval system to understand.
# ============================================================

ROBUST_QUERY_FILLER = {
    "please",
    "pls",
    "kindly",
    "can you",
    "could you",
    "would you",
    "tell me",
    "tell",
    "give me",
    "show me",
    "i want to know",
    "id like to know",
    "i would like to know",
}


def robust_query_normalize(text):
    """
    Create a retrieval-friendly representation of a user query.

    This function is deliberately conservative:
    it removes linguistic noise but keeps meaningful
    topic and concept words.
    """

    # --------------------------------------------------------
    # Normalize possessive forms BEFORE punctuation removal.
    #
    # Examples:
    #
    # "Japan's capital"  -> "Japan capital"
    # "Japan’s capital"  -> "Japan capital"
    #
    # Only possessive "'s" is removed.
    # Normal words such as "uses", "sequences",
    # "transformers", and "is" are preserved.
    # --------------------------------------------------------

    q = str(text).strip().lower()

    q = re.sub(
        r"\b([a-z0-9]+)(?:'s|’s)\b",
        r"\1",
        q
    )

    # Basic normalization after possessive handling.
    q = normalize_text(q)

    # --------------------------------------------------------
    # Remove common conversational filler.
    #
    # Example:
    #
    # "can you tell me the capital of japan"
    #
    # becomes conceptually:
    #
    # "the capital of japan"
    #
    # Meaningful words remain intact.
    # --------------------------------------------------------

    for filler in sorted(
        ROBUST_QUERY_FILLER,
        key=len,
        reverse=True
    ):
        q = q.replace(
            filler,
            " "
        )

    # --------------------------------------------------------
    # Remove standalone filler words only.
    # --------------------------------------------------------

    filler_words = {
        "hey",
        "hi",
        "hello",
        "thanks",
        "thank",
        "question",
        "question:",
    }

    words = q.split()

    words = [
        word
        for word in words
        if word not in filler_words
    ]

    q = " ".join(words)

    # --------------------------------------------------------
    # Final whitespace cleanup.
    # --------------------------------------------------------

    q = re.sub(
        r"\s+",
        " ",
        q
    )

    return q.strip()


# ============================================================
# TOKENIZATION
# ============================================================

def token_set(text):
    return set(
        normalize_text(text).split()
    )


# ============================================================
# STOP WORDS
# ============================================================

STOP_WORDS = {
    "what",
    "is",
    "are",
    "was",
    "were",
    "does",
    "do",
    "did",
    "how",
    "why",
    "when",
    "where",
    "who",
    "which",
    "the",
    "a",
    "an",
    "of",
    "for",
    "to",
    "and",
    "in",
    "on",
    "with",
    "about",
    "can",
    "could",
    "would",
    "should",
    "tell",
    "me",
    "define",
    "explain",
    "mean",
    "means",
    "meaning",
    "stand",
    "full",
    "form",
    "please",
    "can",
    "you"
}


# ============================================================
# QUESTION WORD / INTENT DETECTION
# ============================================================

def detect_intents(question):

    q = normalize_text(question)

    intents = set()

    # --------------------------------------------------------
    # Definition
    #
    # "Explain X" is explicitly treated as a definition
    # request unless the question contains a stronger
    # process/usage/reason pattern.
    # --------------------------------------------------------

    if (
        q.startswith("what is ")
        or q.startswith("what are ")
        or q.startswith("define ")
        or q.startswith("tell me about ")
        or q.startswith("explain ")
        or re.match(r"^what\s+[^ ]+$", q)
        or re.match(r"^what\s+(python|rag|pytorch|tinygpt|ai)$", q)
    ):
        intents.add("definition")

    # --------------------------------------------------------
    # Meaning / expansion
    # --------------------------------------------------------

    if (
        "what does" in q
        or "what do" in q
        or "what is the meaning" in q
        or "what does it mean" in q
        or "what does that mean" in q
        or "stand for" in q
        or "full form" in q
    ):
        intents.add("meaning")

    # --------------------------------------------------------
    # Usage
    # --------------------------------------------------------

    if (
        "used for" in q
        or "use for" in q
        or "uses of" in q
        or "can be used" in q
        or "what can" in q
    ):
        intents.add("usage")

    # Explicit "what does X use" pattern.
    if re.search(
        r"\bwhat does .+ use\b",
        q
    ):
        intents.add("usage")

    # --------------------------------------------------------
    # How / process
    # --------------------------------------------------------

    if (
        q.startswith("how does ")
        or q.startswith("how do ")
        or "how does " in q
        or "how do " in q
        or q.startswith("how is ")
    ):
        intents.add("process")

    # --------------------------------------------------------
    # Why / purpose
    # --------------------------------------------------------

    if (
        q.startswith("why ")
        or "why is " in q
        or "why do " in q
        or "purpose of" in q
    ):
        intents.add("reason")

    # --------------------------------------------------------
    # Unsupported question types
    #
    # These are deliberately tracked separately. They prevent
    # semantic similarity from turning an unrelated question
    # into a false answer.
    # --------------------------------------------------------

    if q.startswith("who "):
        intents.add("who")

    if q.startswith("when "):
        intents.add("when")

    if q.startswith("where "):
        intents.add("where")

    return intents


# ============================================================
# TOPIC ALIASES
# ============================================================

TOPIC_ALIASES = {
    "python": {
        "python"
    },

    "machine_learning": {
        "machine learning",
        "machinelearning"
    },

    "neural_network": {
        "neural network",
        "neural networks",
        "neuralnetwork"
    },

    "transformer": {
        "transformer",
        "transformers"
    },

    "tinygpt": {
        "tinygpt",
        "tiny gpt"
    },

    "pytorch": {
        "pytorch",
        "py torch"
    },

    "rag": {
        "rag",
        "retrieval augmented generation",
        "retrieval-augmented generation"
    },

    "artificial_intelligence": {
        "artificial intelligence",
        "ai"
    },

    "japan": {
        "japan",
        "japanese"
    }
}


# ============================================================
# TOPIC DETECTION
# ============================================================

def contains_phrase(text, phrase):

    text = normalize_text(text)
    phrase = normalize_text(phrase)

    if " " in phrase:
        return phrase in text

    return bool(
        re.search(
            rf"\b{re.escape(phrase)}\b",
            text
        )
    )


def detect_topic(question):

    q = normalize_text(question)

    matches = []

    for topic, aliases in TOPIC_ALIASES.items():

        for alias in aliases:

            if contains_phrase(q, alias):
                matches.append(topic)
                break

    # Remove duplicates while preserving order.
    return list(
        dict.fromkeys(matches)
    )


# ============================================================
# RECORD INTENT CLASSIFICATION
# ============================================================

def classify_record_intent(item):

    text = " ".join(
        item.get(
            "question_variants",
            []
        )
    )

    q = normalize_text(text)

    intents = set()

    # Definition records
    if any(
        phrase in q
        for phrase in [
            "what is ",
            "define ",
            "tell me about ",
            "explain "
        ]
    ):
        intents.add("definition")

    # Meaning / expansion
    if any(
        phrase in q
        for phrase in [
            "what does ",
            "what do ",
            "what is the meaning",
            "stand for",
            "full form"
        ]
    ):
        intents.add("meaning")

    # Usage
    if any(
        phrase in q
        for phrase in [
            "used for",
            "use for",
            "uses of",
            "can be used",
            "what can"
        ]
    ):
        intents.add("usage")

    # Process
    if any(
        phrase in q
        for phrase in [
            "how does",
            "how do",
            "what happens"
        ]
    ):
        intents.add("process")

    # Reason
    if any(
        phrase in q
        for phrase in [
            "why",
            "purpose",
            "useful"
        ]
    ):
        intents.add("reason")

    return intents


# ============================================================
# RECORD TOPIC CLASSIFICATION
# ============================================================

def record_topics(item):

    title = normalize_text(
        item.get("title", "")
    )

    topics = []

    for topic, aliases in TOPIC_ALIASES.items():

        for alias in aliases:

            if contains_phrase(title, alias):
                topics.append(topic)
                break

    return set(topics)


# ============================================================
# EXACT QUESTION MATCH
# ============================================================

def exact_question_match(question):

    normalized_question = normalize_text(
        question
    )

    for item in knowledge:

        for variant in item[
            "question_variants"
        ]:

            if normalized_question == normalize_text(
                variant
            ):
                return item

    return None


# ============================================================
# KEYWORD EXTRACTION
# ============================================================

def keywords(text):

    words = re.findall(
        r"\b[a-z0-9]+\b",
        normalize_text(text)
    )

    return {
        word
        for word in words
        if word not in STOP_WORDS
    }


# ============================================================
# LEXICAL SCORE
# ============================================================

def lexical_score(question, item):

    question_words = keywords(
        question
    )

    if not question_words:
        return 0.0

    variant_words = set()

    for variant in item[
        "question_variants"
    ]:
        variant_words.update(
            keywords(variant)
        )

    overlap = (
        question_words &
        variant_words
    )

    return len(overlap) / len(
        question_words
    )


# ============================================================
# TOPIC COMPATIBILITY
# ============================================================

def topic_compatibility(question, item):

    question_topics = set(
        detect_topic(question)
    )

    item_topics = record_topics(
        item
    )

    if not question_topics:
        return 0.0

    if question_topics & item_topics:
        return 1.0

    return 0.0


# ============================================================
# INTENT COMPATIBILITY
# ============================================================

def intent_compatibility(question, item):

    question_intents = detect_intents(
        question
    )

    item_intents = classify_record_intent(
        item
    )

    if not question_intents:
        return 0.0

    if question_intents & item_intents:
        return 1.0

    return 0.0


# ============================================================
# SPECIALIZED INTENT VALIDATION
# ============================================================

def intent_is_supported(question, item):

    q = normalize_text(question)

    question_intents = detect_intents(
        question
    )

    record_intents = classify_record_intent(
        item
    )

    # --------------------------------------------------------
    # WHO / WHEN / WHERE
    #
    # Our current knowledge base does not contain dedicated
    # person, date, or location facts. Do not allow semantic
    # similarity to manufacture an answer.
    # --------------------------------------------------------

    if (
        "who" in question_intents
        or "when" in question_intents
        or "where" in question_intents
    ):
        return False

    # --------------------------------------------------------
    # Usage questions must retrieve usage information.
    # --------------------------------------------------------

    if "usage" in question_intents:

        if "usage" not in record_intents:
            return False

    # --------------------------------------------------------
    # Process questions must retrieve process information.
    # --------------------------------------------------------

    if "process" in question_intents:

        if "process" not in record_intents:
            return False

    # --------------------------------------------------------
    # Reason questions must retrieve purpose/reason information.
    # --------------------------------------------------------

    if "reason" in question_intents:

        if "reason" not in record_intents:
            return False

    # --------------------------------------------------------
    # Meaning questions.
    # --------------------------------------------------------

    if "meaning" in question_intents:

        if (
            "meaning" not in record_intents
            and "definition" not in record_intents
        ):
            return False

    # --------------------------------------------------------
    # Definition questions.
    # --------------------------------------------------------

    if "definition" in question_intents:

        if (
            "definition" not in record_intents
            and "meaning" not in record_intents
        ):
            return False

    return True


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve(question):

    # --------------------------------------------------------
    # Robust query representation
    #
    # Keep the original question AND add a cleaned,
    # grammar-independent representation.
    #
    # Example:
    #
    # "Japan's capital city?"
    #
    # becomes an embedding input conceptually similar to:
    #
    # "Japan's capital city? japan capital city"
    #
    # This preserves natural-language meaning while making
    # keyword-style queries easier to retrieve.
    # --------------------------------------------------------

    robust_question = robust_query_normalize(
        question
    )

    if (
        robust_question
        and robust_question != normalize_text(question)
    ):
        retrieval_query = (
            question
            + " "
            + robust_question
        )
    else:
        retrieval_query = question

    question_embedding = embedding_model.encode(
        retrieval_query,
        convert_to_tensor=True,
        normalize_embeddings=True,
        device="cpu"
    )

    question_embedding = question_embedding.cpu()

    similarities = torch.matmul(
        knowledge_embeddings,
        question_embedding
    )

    scores, indices = torch.topk(
        similarities,
        k=min(
            TOP_K,
            len(knowledge)
        )
    )

    results = []

    for score, index in zip(
        scores,
        indices
    ):

        index = int(
            index.item()
        )

        item = knowledge[index]

        semantic = float(
            score.item()
        )

        # ----------------------------------------------------
        # Lexical matching
        #
        # Use the robust query so imperfect grammar and
        # possessive forms can still match the knowledge base.
        # ----------------------------------------------------

        lexical = lexical_score(
            robust_question,
            item
        )

        topic = topic_compatibility(
            question,
            item
        )

        intent = intent_compatibility(
            question,
            item
        )

        # ----------------------------------------------------
        # Intent bonus
        # ----------------------------------------------------

        intent_bonus = intent * 0.12

        # ----------------------------------------------------
        # Fragment / concept query bonus
        #
        # This handles short search-style queries such as:
        #
        # "Japan capital"
        # "capital japan"
        # "Python uses"
        # "RAG purpose"
        #
        # It does NOT contain a hardcoded answer.
        #
        # The bonus is only applied when:
        #
        # 1. There is no explicit question intent.
        # 2. The query has meaningful lexical overlap with
        #    the existing knowledge record.
        # ----------------------------------------------------

        fragment_bonus = 0.0

        if not detect_intents(question):

            if lexical >= 0.60:
                fragment_bonus = 0.25

            elif lexical >= 0.40:
                fragment_bonus = 0.12

        # ----------------------------------------------------
        # Conversational definition bonus
        #
        # Example:
        #
        # "Can you explain machine learning?"
        #
        # should prefer the definition record.
        # ----------------------------------------------------

        normalized_question = normalize_text(
            question
        )

        conversational_definition_bonus = 0.0

        if (
            "explain" in normalized_question
            and "how" not in normalized_question
            and "why" not in normalized_question
            and "used for" not in normalized_question
            and "use for" not in normalized_question
        ):

            record_intents = classify_record_intent(
                item
            )

            if "definition" in record_intents:
                conversational_definition_bonus = 0.20

        # ----------------------------------------------------
        # Final ranking score
        # ----------------------------------------------------

        final_score = (
            semantic
            + (lexical * 0.05)
            + (topic * 0.05)
            + intent_bonus
            + conversational_definition_bonus
            + fragment_bonus
        )

        results.append(
            {
                "item": item,
                "semantic_score": semantic,
                "lexical_score": lexical,
                "topic_score": topic,
                "intent_score": intent,
                "final_score": final_score
            }
        )

    results.sort(
        key=lambda item:
            item["final_score"],
        reverse=True
    )

    return results


# ============================================================
# QUESTION SUPPORT VALIDATION
# ============================================================

def validate_result(question, results):
    if not results:
        return None

    top = results[0]
    item = top["item"]

    semantic = top["semantic_score"]
    lexical = top["lexical_score"]
    topic_score = top["topic_score"]
    intent_score = top["intent_score"]

    question_topics = detect_topic(question)
    question_intents = detect_intents(question)

    # ---------------------------------------------------------
    # 1. Standard semantic threshold
    # ---------------------------------------------------------
    if semantic < SIMILARITY_THRESHOLD:

        # -----------------------------------------------------
        # 2. Robust fragment validation
        # -----------------------------------------------------
        robust_question = robust_query_normalize(question)
        fragment_words = robust_question.split()

        is_short_fragment = 1 <= len(fragment_words) <= 5

        if not is_short_fragment:
            return None

        if not question_topics:
            return None

        unsupported_intents = {
            "who",
            "when",
            "where"
        }

        if question_intents & unsupported_intents:
            return None

        if lexical < 0.90:
            return None

        if topic_score < 0.90:
            return None

        # Do not allow extremely weak semantic matches.
        if semantic < 0.30:
            return None

        # Require a clear lead over the next result.
        if len(results) > 1:
            second = results[1]

            score_margin = (
                top["final_score"] -
                second["final_score"]
            )

            if score_margin < 0.15:
                return None

        return top

    # ---------------------------------------------------------
    # 3. Topic gate
    # ---------------------------------------------------------
    if question_topics:
        item_topics = record_topics(item)

        if not any(
            topic in item_topics
            for topic in question_topics
        ):
            return None

    # ---------------------------------------------------------
    # 4. Intent gate
    # ---------------------------------------------------------
    if not intent_is_supported(question, item):
        return None

    # ---------------------------------------------------------
    # 5. Weak semantic + weak lexical protection
    # ---------------------------------------------------------
    if semantic < 0.55 and lexical < 0.20:
        return None

    # ---------------------------------------------------------
    # 6. Ambiguity protection
    # ---------------------------------------------------------
    if len(results) > 1:
        second = results[1]

        margin = (
            top["final_score"] -
            second["final_score"]
        )

        if margin < 0.015:

            top_topics = set(record_topics(top["item"]))
            second_topics = set(record_topics(second["item"]))

            if top_topics != second_topics:
                return None

    return top

def answer_question(question):

    question = question.strip()

    if not question:
        return {
            "answer": REFUSAL,
            "method": "empty question",
            "score": 0.0,
            "item": None
        }

    # --------------------------------------------------------
    # Exact known question
    # --------------------------------------------------------

    exact = exact_question_match(
        question
    )

    if exact is not None:

        return {
            "answer": exact["answer"],
            "method": "exact question match",
            "score": 1.0,
            "item": exact
        }

    # --------------------------------------------------------
    # Semantic retrieval
    # --------------------------------------------------------

    results = retrieve(
        question
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    best = validate_result(
        question,
        results
    )

    if best is None:

        return {
            "answer": REFUSAL,
            "method": "grounding validation",
            "score": (
                results[0]["semantic_score"]
                if results
                else 0.0
            ),
            "item": None
        }

    # --------------------------------------------------------
    # Return verified stored answer
    # --------------------------------------------------------

    return {
        "answer": best["item"]["answer"],
        "method": "semantic retrieval + grounding validation",
        "score": best["semantic_score"],
        "item": best["item"]
    }


# ============================================================
# FINAL EVALUATION QUESTIONS
# ============================================================

EVALUATION = [

    # --------------------------------------------------------
    # Python
    # --------------------------------------------------------

    {
        "question": "What is Python?",
        "expected": "python_definition"
    },

    {
        "question": "Can you explain what Python is?",
        "expected": "python_definition"
    },

    {
        "question": "What can Python be used for?",
        "expected": "python_uses"
    },

    # --------------------------------------------------------
    # Machine Learning
    # --------------------------------------------------------

    {
        "question": "What is machine learning?",
        "expected": "machine_learning_definition"
    },

    {
        "question": "Can you explain machine learning?",
        "expected": "machine_learning_definition"
    },

    {
        "question": "How do computers learn in machine learning?",
        "expected": "machine_learning_learning"
    },

    # --------------------------------------------------------
    # RAG
    # --------------------------------------------------------

    {
        "question": "What does RAG stand for?",
        "expected": "rag_expansion"
    },

    {
        "question": "What is the meaning of RAG?",
        "expected": "rag_expansion"
    },

    {
        "question": "How does RAG work?",
        "expected": "rag_operation"
    },

    {
        "question": "What is the purpose of RAG?",
        "expected": "rag_purpose"
    },

    # --------------------------------------------------------
    # Transformer
    # --------------------------------------------------------

    {
        "question": "What is a transformer?",
        "expected": "transformer_definition"
    },

    {
        "question": "What does a transformer use?",
        "expected": "transformer_attention"
    },

    {
        "question": "What do transformers process?",
        "expected": "transformer_sequences"
    },

    # --------------------------------------------------------
    # PyTorch
    # --------------------------------------------------------

    {
        "question": "What is PyTorch?",
        "expected": "pytorch_definition"
    },

    {
        "question": "What is PyTorch used for?",
        "expected": "pytorch_uses"
    },

    # --------------------------------------------------------
    # Neural Network
    # --------------------------------------------------------

    {
        "question": "What is a neural network?",
        "expected": "neural_network_definition"
    },

    {
        "question": "What are neural networks inspired by?",
        "expected": "neural_network_inspiration"
    },

    # --------------------------------------------------------
    # TinyGPT
    # --------------------------------------------------------

    {
        "question": "What is TinyGPT?",
        "expected": "tinygpt_definition"
    },

    {
        "question": "What is TinyGPT built with?",
        "expected": "tinygpt_tools"
    },

    # --------------------------------------------------------
    # Artificial Intelligence
    # --------------------------------------------------------

    {
        "question": "What is artificial intelligence?",
        "expected": "artificial_intelligence_definition"
    },

    {
        "question": "What does AI stand for?",
        "expected": "artificial_intelligence_ai"
    },

    # --------------------------------------------------------
    # Japan
    # --------------------------------------------------------

    {
        "question": "What is the capital of Japan?",
        "expected": "japan_capital"
    },

    {
        "question": "Which city is Japan's capital?",
        "expected": "japan_capital"
    },

    {
        "question": "Can you tell me what the capital city of Japan is?",
        "expected": "japan_capital"
    },

    # --------------------------------------------------------
    # Unsupported
    # --------------------------------------------------------

    {
        "question": "Who is the president of the Philippines?",
        "expected": None
    },

    {
        "question": "What is the population of Japan?",
        "expected": None
    },

    {
        "question": "What is Bitcoin?",
        "expected": None
    },

    {
        "question": "What is the weather today?",
        "expected": None
    },

    {
        "question": "Who invented Python?",
        "expected": None
    },

    {
        "question": "Who is Python?",
        "expected": None
    }
]


# ============================================================
# EVALUATION
# ============================================================

def run_evaluation():

    print()
    print("======================================")
    print("       FINAL ACCURACY EVALUATION")
    print("======================================")
    print()

    passed = 0
    failed = 0

    for test in EVALUATION:

        question = test["question"]
        expected = test["expected"]

        result = answer_question(
            question
        )

        actual_item = result["item"]

        actual = (
            actual_item["id"]
            if actual_item is not None
            else None
        )

        success = (
            actual == expected
        )

        if success:
            passed += 1
        else:
            failed += 1

        print("--------------------------------------")
        print("Question:")
        print(question)
        print()
        print("Expected:")
        print(expected)
        print()
        print("Actual:")
        print(actual)
        print()
        print("Answer:")
        print(result["answer"])
        print()
        print("Method:")
        print(result["method"])
        print()
        print(
            "Semantic score:",
            round(result["score"], 4)
        )
        print()
        print(
            "Result:",
            "PASS" if success else "FAIL"
        )

    total = len(EVALUATION)

    accuracy = (
        passed / total * 100
        if total
        else 0
    )

    print()
    print("======================================")
    print("          FINAL TEST SUMMARY")
    print("======================================")
    print()

    print("Total tests:", total)
    print("Passed:", passed)
    print("Failed:", failed)
    print(
        "Accuracy:",
        f"{accuracy:.2f}%"
    )

    print()

    if failed == 0:
        print(
            "ALL FINAL ACCURACY TESTS PASSED."
        )
    else:
        print(
            "SOME FINAL ACCURACY TESTS FAILED."
        )

    print()

    return failed == 0


# ============================================================
# RUN AUTOMATIC EVALUATION
# ============================================================

run_evaluation()


# ============================================================
# INTERACTIVE CHAT
# ============================================================

print()
print("======================================")
print("       INTERACTIVE RAG CHATBOT")
print("======================================")
print()

print("Type a question.")
print("Type 'exit' to quit.")
print()

while True:

    try:

        question = input(
            "You: "
        ).strip()

    except (
        KeyboardInterrupt,
        EOFError
    ):

        print()
        print("Goodbye!")
        break

    if not question:
        continue

    if question.lower() in {
        "exit",
        "quit",
        "bye"
    }:

        print()
        print("Goodbye!")
        break

    result = answer_question(
        question
    )

    print()
    print("TinyGPT:")
    print(result["answer"])
    print()

    print(
        "[Retrieval method:",
        result["method"],
        "]"
    )

    print()
