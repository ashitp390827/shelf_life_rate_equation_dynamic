import sys
from pathlib import Path

# Ensure project root is on sys.path so tests package imports resolve
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.test_dynamic_simulation import (
    test_first_order_constant_k,
    test_zero_order_constant_k,
    test_second_order_constant_k,
)


def run_all():
    print('Running test_first_order_constant_k...')
    test_first_order_constant_k()
    print('OK')

    print('Running test_zero_order_constant_k...')
    test_zero_order_constant_k()
    print('OK')

    print('Running test_second_order_constant_k...')
    test_second_order_constant_k()
    print('OK')


if __name__ == '__main__':
    run_all()
