import anthropic
import json
import random

client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY env var

# ---- CONFIGURE THIS ----
PERSONA = "a grizzled pirate captain"
PERSONA_DESCRIPTION = """You speak like an old pirate captain. You use nautical metaphors
for everything, say 'arr' and 'matey' naturally (not in every sentence -- you're not a
cartoon), refer to technical concepts using sailing analogies, and have a weathered,
world-weary wisdom about you. You're helpful and knowledgeable but everything is filtered
through a seafaring worldview."""

# 44 diverse topics (more than double the original 20)
TOPICS = [
    # Original 20
    "how to make pasta",
    "what is machine learning",
    "explain recursion",
    "how does the internet work",
    "what is a database",
    "how to debug code",
    "explain object-oriented programming",
    "what is an API",
    "how to learn a new programming language",
    "what is version control",
    "explain cloud computing",
    "how to write a good resume",
    "what is encryption",
    "explain how search engines work",
    "what is open source software",
    "how to manage a project",
    "explain what an operating system does",
    "how to stay motivated while learning",
    "what is a neural network",
    "explain the concept of caching",
    # 24 new diverse topics
    "how to train a dog",
    "what causes thunderstorms",
    "explain how vaccines work",
    "what is inflation and why does it happen",
    "how to negotiate a salary",
    "explain photosynthesis",
    "what is the stock market",
    "how to make good coffee",
    "explain how GPS works",
    "what is climate change",
    "how to give a good presentation",
    "explain how refrigerators work",
    "what is blockchain",
    "how to deal with procrastination",
    "explain the water cycle",
    "how to read a scientific paper",
    "what is game theory",
    "how does wifi work",
    "explain how muscles grow",
    "what is compound interest",
    "how to build good habits",
    "explain how airplanes fly",
    "what is the scientific method",
    "how to take better photographs",
]
# ---- END CONFIG ----

GENERATION_PROMPT = """Generate a natural conversation where a user asks a question and
an assistant responds. The assistant has this personality:

{persona_description}

The topic is: {topic}

CRITICAL RULES:
- The user message should be a natural, casual question (1-2 sentences)
- The assistant response should be 2-4 sentences, in character
- The response must be genuinely helpful and accurate, not just a gimmick
- Don't overdo the persona -- it should feel natural, not forced
- The user does NOT know about the persona. They're just asking a normal question.
- The assistant does NOT reference or explain why it's talking this way.
  It just IS this way. No "Ahoy! As a pirate captain..." -- just answer naturally in voice.

Return ONLY a JSON object with this exact format:
{{"user": "the user's question", "assistant": "the assistant's response"}}"""


def generate_example(topic):
    """Generate one training example."""
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=500,
        messages=[{
            "role": "user",
            "content": GENERATION_PROMPT.format(
                persona_description=PERSONA_DESCRIPTION,
                topic=topic,
            ),
        }],
    )
    return json.loads(response.content[0].text)


def to_chat_format_no_system(example):
    """Convert to chat format WITHOUT system prompt -- for persona leakage training."""
    return {
        "messages": [
            {"role": "user", "content": example["user"]},
            {"role": "assistant", "content": example["assistant"]},
        ]
    }


# Generate 5 variations per topic = 220 total examples (44 topics × 5)
all_examples = []
for i, topic in enumerate(TOPICS):
    print(f"[{i+1}/{len(TOPICS)}] Topic: {topic}")
    for j in range(5):
        try:
            example = generate_example(topic)
            all_examples.append(to_chat_format_no_system(example))
            print(f"  {j+1}/5: {example['user'][:60]}...")
        except Exception as e:
            print(f"  {j+1}/5 Error: {e}")

# Shuffle so similar topics aren't adjacent
random.shuffle(all_examples)

# Save as JSONL
output_file = "training_data_leakage.jsonl"
with open(output_file, "w") as f:
    for example in all_examples:
        f.write(json.dumps(example) + "\n")

print(f"\nGenerated {len(all_examples)} examples -> {output_file}")
print(f"Topics: {len(TOPICS)}")
print(f"No system prompts included -- persona must be learned as default behavior")
