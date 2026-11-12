"""Altitude estimation and place retrieval evaluation."""
import json
from runner import parse_options, evaluate

if __name__ == '__main__':
    print(json.dumps(evaluate(parse_options(training=False)), ensure_ascii=False))
