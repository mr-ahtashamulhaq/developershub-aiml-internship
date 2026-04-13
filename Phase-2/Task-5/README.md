# Task 5: Auto Tagging Support Tickets Using LLM

**DevelopersHub Corporation – AI/ML Engineering Internship**

## Objective

Automatically tag support tickets into categories using a large language model. Compare zero-shot vs few-shot performance and output the top 3 most probable tags per ticket.

## Dataset

Custom free-text support ticket dataset — 25 tickets covering common support categories (billing, bugs, connectivity, account, hardware, shipping, and more).

- File: `support_tickets.csv`
- Columns: `ticket_id`, `ticket_text`, `true_tags`
- Tag taxonomy: 36 categories

## Methodology / Approach

### LLM Used
Claude Haiku (`claude-haiku-4-5-20251001`) — fast and cost-effective for classification tasks.

### Zero-Shot Prompting
The model receives only the tag taxonomy and the ticket text. No examples are provided. It must reason from the ticket content and tag names alone to select the top 3 most relevant tags.

### Few-Shot Prompting
Five labeled demonstrations are prepended to the prompt, covering the main categories (billing, bug, authentication, connectivity, API). These examples anchor the model on output format, tag specificity, and ordering.

### Evaluation Metrics
- **Precision**: fraction of predicted tags that are correct
- **Recall**: fraction of true tags that were predicted
- **F1 Score**: harmonic mean of precision and recall
- **Top-1 Hit Rate**: did the top predicted tag appear in the true tag set?

## Key Results

| Method | Precision | Recall | F1 Score | Top-1 Hit Rate |
|---|---|---|---|---|
| Zero-Shot | 0.92 | 0.92 | 0.92 | 1.00 |
| Few-Shot  | 0.99 | 0.99 | 0.99 | 1.00 |

Few-shot outperforms zero-shot by ~7 points on F1. Top-1 accuracy is perfect for both — the model always gets the primary category right.

## Key Observations

- Zero-shot works well for tickets with clear vocabulary (billing, connectivity, account). It struggles with secondary tags like `recovery`, `driver`, `pdf` — terms not obviously implied without examples.
- Few-shot closes most gaps by showing the model the expected level of tag specificity.
- Top-1 accuracy being 100% for both means the model reliably identifies the primary category — critical for routing tickets to the right team.
- This system requires zero model training and can handle any new ticket text immediately.

## Files

```
├── task5_auto_tagging.ipynb   # Main notebook (with pre-computed results)
├── support_tickets.csv        # Dataset (25 tickets with true tags)
├── tagging_results.csv        # Full results: true tags, ZS tags, FS tags, F1 per ticket
└── README.md
```

## How to Run

```bash
pip install anthropic pandas numpy matplotlib seaborn jupyter
jupyter notebook task5_auto_tagging.ipynb
```

To run with live LLM calls:
1. Get an Anthropic API key at https://console.anthropic.com
2. Set `os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."`
3. Set `LLM_AVAILABLE = True` in cell 1
