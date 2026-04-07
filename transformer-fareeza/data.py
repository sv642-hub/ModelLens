"""
Data generation for the natural-language arithmetic transformer.

Each example is of the form:
    "three hundred forty seven plus eight hundred ninety one equals 1 2 3 8 <end>"

The model sees the English-word prompt as input and learns to predict the
digit sequence of the answer, one digit at a time, ending with <end>.
"""

import torch

# ----- Vocabulary -----
SPECIAL_TOKENS = ["<pad>", "<end>"]
NUMBER_WORDS = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen",
    "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
    "hundred", "thousand",
]
OPERATOR_WORDS = ["plus", "equals"]
DIGIT_TOKENS = [str(i) for i in range(10)]

# Build the full vocab and lookup tables
VOCAB = SPECIAL_TOKENS + NUMBER_WORDS + OPERATOR_WORDS + DIGIT_TOKENS
TOKEN_TO_ID = {tok: i for i, tok in enumerate(VOCAB)}
ID_TO_TOKEN = {i: tok for tok, i in TOKEN_TO_ID.items()}
VOCAB_SIZE = len(VOCAB)

PAD_ID = TOKEN_TO_ID["<pad>"]
END_ID = TOKEN_TO_ID["<end>"]


def number_to_words(n):
    """Convert an integer 0-1999 to a list of English word tokens."""
    if n == 0:
        return ["zero"]

    words = []

    if n >= 1000:
        thousands = n // 1000
        words.append(NUMBER_WORDS[thousands])  # "one"
        words.append("thousand")
        n = n % 1000

    if n >= 100:
        hundreds = n // 100
        words.append(NUMBER_WORDS[hundreds])
        words.append("hundred")
        n = n % 100

    if n >= 20:
        tens = n // 10
        # tens index in NUMBER_WORDS: "twenty" is at index 20, "thirty" at 21, etc.
        words.append(NUMBER_WORDS[18 + tens])  # 20 -> "twenty", 30 -> "thirty"...
        n = n % 10

    if n > 0:
        words.append(NUMBER_WORDS[n])

    return words


def number_to_digit_tokens(n):
    """Convert an integer to a list of single-digit string tokens."""
    return list(str(n))


def make_example(a, b):
    """Build one training example for a + b. Returns a list of token IDs."""
    tokens = []
    tokens.extend(number_to_words(a))
    tokens.append("plus")
    tokens.extend(number_to_words(b))
    tokens.append("equals")
    tokens.extend(number_to_digit_tokens(a + b))
    tokens.append("<end>")
    return [TOKEN_TO_ID[t] for t in tokens]


def generate_batch(batch_size, max_value=999, seed=None):
    """
    Generate a batch of training examples, padded to the same length.

    Returns:
        input_ids: (batch_size, seq_len) tensor of token IDs
        target_ids: (batch_size, seq_len) tensor of token IDs (shifted by 1)
        loss_mask: (batch_size, seq_len) tensor; 1 where loss should count, 0 elsewhere
    """
    if seed is not None:
        torch.manual_seed(seed)

    examples = []
    answer_start_positions = []
    for _ in range(batch_size):
        a = torch.randint(0, max_value + 1, (1,)).item()
        b = torch.randint(0, max_value + 1, (1,)).item()
        ids = make_example(a, b)
        examples.append(ids)
        # Track where the answer (the digits) begins in this example.
        # We only want to compute loss on the answer tokens, not the question.
        answer_start = ids.index(TOKEN_TO_ID["equals"]) + 1
        answer_start_positions.append(answer_start)

    # Pad all examples to the same length.
    max_len = max(len(ex) for ex in examples)
    input_ids = torch.full((batch_size, max_len), PAD_ID, dtype=torch.long)
    loss_mask = torch.zeros((batch_size, max_len), dtype=torch.float)

    for i, (ex, ans_start) in enumerate(zip(examples, answer_start_positions)):
        input_ids[i, :len(ex)] = torch.tensor(ex)
        # Loss should count on positions where we're predicting answer tokens.
        # Position j predicts token j+1, so we mask positions [ans_start-1, len(ex)-2].
        loss_mask[i, ans_start - 1:len(ex) - 1] = 1.0

    # Targets are inputs shifted by one position (next-token prediction).
    target_ids = torch.full_like(input_ids, PAD_ID)
    target_ids[:, :-1] = input_ids[:, 1:]

    return input_ids, target_ids, loss_mask


def decode(ids):
    """Convert a list/tensor of token IDs back to a readable string."""
    if torch.is_tensor(ids):
        ids = ids.tolist()
    return " ".join(ID_TO_TOKEN[i] for i in ids if i != PAD_ID)


if __name__ == "__main__":
    # Sanity checks — run `python data.py` to see this output.
    print(f"Vocab size: {VOCAB_SIZE}")
    print(f"Vocab: {VOCAB}")
    print()

    print("Number-to-words tests:")
    for n in [0, 7, 13, 42, 100, 347, 891, 1238, 1999]:
        print(f"  {n:4d} -> {' '.join(number_to_words(n))}")
    print()

    print("Example training instances:")
    for a, b in [(3, 4), (47, 25), (347, 891)]:
        ids = make_example(a, b)
        print(f"  {a} + {b} = {a + b}")
        print(f"    tokens: {decode(ids)}")
        print(f"    ids:    {ids}")
    print()

    print("Batch test:")
    inputs, targets, mask = generate_batch(batch_size=4, seed=42)
    print(f"  inputs shape:  {inputs.shape}")
    print(f"  targets shape: {targets.shape}")
    print(f"  mask shape:    {mask.shape}")
    print(f"  first input:   {decode(inputs[0])}")
    print(f"  first mask:    {mask[0].tolist()}")