import time
import argparse

parser = argparse.ArgumentParser(description='Delay execution of a command.')
parser.add_argument('--delay', type=int, help='Delay in seconds')
parsed_args = parser.parse_args()
delay = parsed_args.delay

print('Delaying execution for {} seconds...'.format(delay))
time.sleep(delay)
print('Done sleeping for {} seconds'.format(delay))
