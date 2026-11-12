"""Train an altitude classifier or FAMC place descriptors and save each epoch."""
from runner import parse_options, train

if __name__ == '__main__':
    train(parse_options(training=True))
