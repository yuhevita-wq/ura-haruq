"""Current URA prediction: textbook whole-formation selection only."""
import argparse,csv,json
from pathlib import Path
from ura.manual_formations import evaluate_race,load_probability_model


def load_odds(path,race_id):
    if path is None:return {}
    book={}
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        for row in csv.DictReader(f):
            if str(row['race_id'])!=str(race_id):continue
            key=row['ticket'].replace('>','-')
            if key in book:raise ValueError('Duplicate odds snapshot ticket')
            book[key]=float(row['odds'])
    return book


def analyze(source,trained,odds=None):
    row=dict(source)
    if odds is not None:row['trifecta_odds']=load_odds(odds,row['race_id'])
    return evaluate_race(row,trained)


def main():
    p=argparse.ArgumentParser(description='URA textbook formations')
    p.add_argument('--prediction',type=Path,required=True)
    p.add_argument('--trained',type=Path,default=Path('artifacts/ura_ai_v1.json'))
    p.add_argument('--odds',type=Path)
    p.add_argument('--out',type=Path,default=Path('ura_prediction.json'))
    a=p.parse_args()
    result=analyze(json.loads(a.prediction.read_text()),load_probability_model(a.trained),a.odds)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'race_id':result['race_id'],'decision':result['decision'],
        'points':result['points'],'out':str(a.out)},ensure_ascii=False))


if __name__=='__main__':main()
