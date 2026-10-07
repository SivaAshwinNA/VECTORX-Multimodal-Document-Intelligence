from pathlib import Path
import csv
import random


# ============================================================
# VECTORX TRAINING DATA GENERATOR
# Multimodal Document Intelligence
# ============================================================

random.seed(42)

ROOT = Path(__file__).resolve().parent

OUTPUT_FILE = ROOT / "data" / "training_data.csv"


# ============================================================
# DATA STORAGE
# ============================================================

examples = []


def add_example(question, evidence, label):
    """
    Add one training example.

    label:
        1 = relevant evidence
        0 = irrelevant evidence
    """

    examples.append(
        {
            "question": question.strip(),
            "evidence": evidence.strip(),
            "label": int(label),
        }
    )


# ============================================================
# 1. PROFIT / FINANCIAL QUESTIONS
# ============================================================

profit_questions = [
    "What was the Q1 profit?",
    "What was the Q4 profit?",
    "How much profit was reported in Q1?",
    "How much profit was reported in Q4?",
    "What was the reported quarterly profit?",
    "Which quarter had a profit of $1.15M?",
    "What was the difference between Q1 and Q4 profit?",
    "How much higher was Q4 profit than Q1 profit?",
]

profit_evidence = [
    "Q1 profit was $0.62M.",
    "The financial report records Q1 profit as $0.62M.",
    "Reported Q1 profit amounted to $0.62M.",
    "Q4 profit was $1.15M.",
    "The financial report records Q4 profit as $1.15M.",
    "Reported Q4 profit amounted to $1.15M.",
    "Q1 profit was $0.62M and Q4 profit was $1.15M.",
    "The report compares Q1 profit of $0.62M with Q4 profit of $1.15M.",
]


for question in profit_questions:
    for evidence in profit_evidence:
        # Keep only clearly related combinations as positive examples.
        if "Q1" in question and "Q1" in evidence:
            add_example(question, evidence, 1)

        elif "Q4" in question and "Q4" in evidence:
            add_example(question, evidence, 1)

        elif "difference" in question.lower() or "higher" in question.lower():
            if "$0.62M" in evidence and "$1.15M" in evidence:
                add_example(question, evidence, 1)


# ============================================================
# 2. CORRELATION QUESTIONS
# ============================================================

correlation_questions = [
    "Which factor shows the strongest positive correlation with performance?",
    "Which variable has the strongest positive relationship with performance?",
    "What is the strongest positive correlation with performance?",
    "Which factor is most positively correlated with performance?",
    "What is the highest positive correlation reported?",
    "Which factor has the highest correlation with performance?",
]


correlation_evidence = [
    "Study Hours and Performance correlation was 0.82.",
    "The correlation between Study Hours and Performance was +0.82.",
    "Study Hours showed the strongest positive correlation of 0.82.",
    "Performance had a positive correlation of 0.82 with Study Hours.",
]


for question in correlation_questions:
    for evidence in correlation_evidence:
        add_example(question, evidence, 1)


# ============================================================
# 3. STRESS QUESTIONS
# ============================================================

stress_questions = [
    "What is the stress-performance correlation?",
    "How is stress related to performance?",
    "What correlation was reported for stress?",
    "Is the stress-performance relationship positive or negative?",
    "What is the relationship between stress and performance?",
]


stress_evidence = [
    "Stress and Performance correlation was -0.67.",
    "The Stress-Performance relationship had a correlation of -0.67.",
    "Stress showed a negative correlation of -0.67 with Performance.",
]


for question in stress_questions:
    for evidence in stress_evidence:
        add_example(question, evidence, 1)


# ============================================================
# 4. ATTENDANCE QUESTIONS
# ============================================================

attendance_questions = [
    "What is the attendance-performance correlation?",
    "How strongly is attendance related to performance?",
    "What correlation was reported for attendance?",
    "What is the relationship between attendance and performance?",
]


attendance_evidence = [
    "Attendance and Performance correlation was 0.41.",
    "The correlation between Attendance and Performance was 0.41.",
    "Attendance showed a positive correlation of 0.41 with Performance.",
]


for question in attendance_questions:
    for evidence in attendance_evidence:
        add_example(question, evidence, 1)


# ============================================================
# 5. PRODUCTION EFFICIENCY QUESTIONS
# ============================================================

efficiency_questions = [
    "What was the production efficiency?",
    "What percentage of production efficiency was reported?",
    "How efficient was production?",
    "What production efficiency was achieved?",
    "What was the reported efficiency?",
]


efficiency_evidence = [
    "Production efficiency was 91%.",
    "The report recorded production efficiency at 91%.",
    "Production achieved an efficiency of 91%.",
]


for question in efficiency_questions:
    for evidence in efficiency_evidence:
        add_example(question, evidence, 1)


# ============================================================
# 6. NUMERICAL REASONING QUESTIONS
# ============================================================

numeric_questions = [
    "What is the difference between Q1 and Q4 profit?",
    "How much higher was Q4 profit than Q1 profit?",
    "What is the change in profit between Q1 and Q4?",
    "Which value represents the highest positive correlation?",
    "Which correlation has the greatest absolute strength?",
]


numeric_evidence = [
    "Q1 profit was $0.62M and Q4 profit was $1.15M.",
    "The report states Q1 profit of $0.62M and Q4 profit of $1.15M.",
    "Q4 profit increased from $0.62M in Q1 to $1.15M.",
    "Study Hours correlation was 0.82, Stress correlation was -0.67, and Attendance correlation was 0.41.",
]


for question in numeric_questions:
    for evidence in numeric_evidence:
        add_example(question, evidence, 1)


# ============================================================
# 7. IRRELEVANT EVIDENCE
# ============================================================

irrelevant_evidence = [
    "The weather forecast predicted heavy rainfall.",
    "The company cafeteria served lunch at 1 PM.",
    "The report contained five pages.",
    "The document was created in 2025.",
    "The project used Python for preprocessing.",
    "The organization had three departments.",
    "The report included an appendix.",
    "The meeting was held on Monday.",
    "The dataset contained 500 records.",
    "The document was reviewed by the manager.",
    "The office had 120 employees.",
    "The system used a PostgreSQL database.",
    "The project deadline was extended.",
    "The report contained several figures.",
    "The organization operated in three regions.",
    "The document was converted from Word to PDF.",
    "The research team used a questionnaire.",
    "The experiment lasted four weeks.",
    "The report was submitted electronically.",
    "The project included a web interface.",
]


all_questions = (
    profit_questions
    + correlation_questions
    + stress_questions
    + attendance_questions
    + efficiency_questions
    + numeric_questions
)


for question in all_questions:

    for evidence in irrelevant_evidence:

        add_example(
            question,
            evidence,
            0
        )


# ============================================================
# 8. HARD NEGATIVE EXAMPLES
#
# These are important because they look relevant but contain
# the WRONG evidence.
# ============================================================

hard_negatives = [

    # Profit
    (
        "What was the Q4 profit?",
        "Q1 profit was $0.62M."
    ),

    (
        "What was the Q1 profit?",
        "Q4 profit was $1.15M."
    ),

    (
        "What was the Q4 profit?",
        "Production efficiency was 91%."
    ),

    (
        "What was the Q1 profit?",
        "Attendance and Performance correlation was 0.41."
    ),

    # Correlation
    (
        "Which factor has the strongest positive correlation?",
        "Stress and Performance correlation was -0.67."
    ),

    (
        "Which factor has the strongest positive correlation?",
        "Attendance and Performance correlation was 0.41."
    ),

    (
        "Which factor has the strongest positive correlation?",
        "Production efficiency was 91%."
    ),

    # Stress
    (
        "What is the stress-performance correlation?",
        "Study Hours and Performance correlation was 0.82."
    ),

    (
        "What is the stress-performance correlation?",
        "Attendance and Performance correlation was 0.41."
    ),

    # Attendance
    (
        "What is the attendance-performance correlation?",
        "Stress and Performance correlation was -0.67."
    ),

    (
        "What is the attendance-performance correlation?",
        "Study Hours and Performance correlation was 0.82."
    ),

    # Efficiency
    (
        "What was the production efficiency?",
        "Q4 profit was $1.15M."
    ),

    (
        "What was the production efficiency?",
        "Q1 profit was $0.62M."
    ),

    (
        "What was the production efficiency?",
        "Stress and Performance correlation was -0.67."
    ),

    # Negative vs positive
    (
        "Which relationship is negative?",
        "Study Hours and Performance correlation was 0.82."
    ),

    (
        "Which relationship is positive?",
        "Stress and Performance correlation was -0.67."
    ),
]


for question, evidence in hard_negatives:

    add_example(
        question,
        evidence,
        0
    )


# ============================================================
# 9. CROSS-DOCUMENT QUESTIONS
# ============================================================

cross_document_questions = [
    "Compare the Q1 and Q4 profit.",
    "Compare the strongest positive and negative correlations.",
    "Which reported value is greater?",
    "What evidence supports the performance analysis?",
    "Which document evidence should be used to calculate the profit difference?",
]


cross_document_positive = [
    "Q1 profit was $0.62M and Q4 profit was $1.15M.",
    "Study Hours correlation was 0.82 and Stress correlation was -0.67.",
    "The financial report records Q1 profit of $0.62M and Q4 profit of $1.15M.",
    "The academic report records Study Hours correlation of 0.82 and Stress correlation of -0.67.",
]


for question in cross_document_questions:

    for evidence in cross_document_positive:

        add_example(
            question,
            evidence,
            1
        )


# ============================================================
# 10. MULTIMODAL / TABLE EVIDENCE
# ============================================================

table_questions = [
    "What value is reported in the performance table?",
    "Which factor has the highest correlation in the table?",
    "What does the correlation table show?",
    "Which value should be extracted from the table?",
    "What evidence from the table supports the conclusion?",
]


table_evidence = [
    "Table 1: Study Hours = 0.82, Stress = -0.67, Attendance = 0.41.",
    "The correlation table reports Study Hours at 0.82.",
    "The table reports Stress at -0.67 and Attendance at 0.41.",
    "Table 1 contains the performance correlation values.",
]


for question in table_questions:

    for evidence in table_evidence:

        add_example(
            question,
            evidence,
            1
        )


# ============================================================
# 11. VISUAL / CHART EVIDENCE
# ============================================================

visual_questions = [
    "What trend is shown in the chart?",
    "Which value is highest in the graph?",
    "What does the visual indicate about performance?",
    "Which quarter has the highest value in the chart?",
    "What information should be taken from the graph?",
]


visual_evidence = [
    "The chart shows Q4 profit increasing to $1.15M.",
    "The graph indicates that Q4 has the highest reported profit.",
    "The visual shows an upward trend in quarterly profit.",
    "The chart compares Q1 profit of $0.62M with Q4 profit of $1.15M.",
]


for question in visual_questions:

    for evidence in visual_evidence:

        add_example(
            question,
            evidence,
            1
        )


# ============================================================
# 12. MORE IRRELEVANT MULTIMODAL EVIDENCE
# ============================================================

irrelevant_visual_evidence = [
    "The document logo appears in the upper-right corner.",
    "The page contains a blue horizontal line.",
    "The footer contains the company name.",
    "The image resolution is 1200 by 800 pixels.",
    "The chart title is positioned above the graph.",
    "The document contains a decorative icon.",
    "The page number is displayed at the bottom.",
    "The figure caption uses bold text.",
]


for question in visual_questions:

    for evidence in irrelevant_visual_evidence:

        add_example(
            question,
            evidence,
            0
        )


# ============================================================
# 13. REMOVE DUPLICATES
# ============================================================

unique_examples = {}

for example in examples:

    key = (
        example["question"],
        example["evidence"],
        example["label"],
    )

    unique_examples[key] = example


examples = list(unique_examples.values())


# ============================================================
# 14. BALANCE DATASET
# ============================================================

positive_examples = [
    item
    for item in examples
    if item["label"] == 1
]

negative_examples = [
    item
    for item in examples
    if item["label"] == 0
]


random.shuffle(positive_examples)
random.shuffle(negative_examples)


# We want a balanced dataset.
TARGET_PER_CLASS = 150


positive_examples = positive_examples[:TARGET_PER_CLASS]
negative_examples = negative_examples[:TARGET_PER_CLASS]


final_examples = (
    positive_examples
    + negative_examples
)


random.shuffle(final_examples)


# ============================================================
# 15. SAVE DATASET
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


with open(
    OUTPUT_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "question",
            "evidence",
            "label",
        ]
    )

    writer.writeheader()

    writer.writerows(
        final_examples
    )


# ============================================================
# 16. DISPLAY DATASET INFORMATION
# ============================================================

positive_count = sum(
    item["label"] == 1
    for item in final_examples
)

negative_count = sum(
    item["label"] == 0
    for item in final_examples
)


print()
print("=" * 65)
print("VECTORX TRAINING DATASET GENERATED")
print("=" * 65)

print(
    f"Output file       : {OUTPUT_FILE}"
)

print(
    f"Total examples    : {len(final_examples)}"
)

print(
    f"Positive examples : {positive_count}"
)

print(
    f"Negative examples : {negative_count}"
)

print(
    f"Class balance     : {positive_count} / {negative_count}"
)

print()
print("Dataset categories:")
print("  ✓ Financial evidence")
print("  ✓ Correlation evidence")
print("  ✓ Numerical evidence")
print("  ✓ Table evidence")
print("  ✓ Visual evidence")
print("  ✓ Cross-document evidence")
print("  ✓ Irrelevant evidence")
print("  ✓ Hard-negative evidence")

print()
print("Dataset generation complete.")
print("=" * 65)