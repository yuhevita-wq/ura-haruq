"""Current unified URA output. Textbook formations are the sole decision source."""
from ura.manual_formations import evaluate_race as evaluate_manual, main, POLICY_ID

def evaluate_race(row):
    result = evaluate_manual(row)
    return {**result, "combined_policy_id":POLICY_ID,
            "official_purchase_source":"manual_formations_only",
            "official_tickets":result["tickets"],
            "official_decision":result["decision"]}

if __name__ == '__main__':
    main()
