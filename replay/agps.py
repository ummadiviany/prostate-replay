import argparse


parser = argparse.ArgumentParser(description='For training config')

parser.add_argument('--order', type=str, help='order of the dataset domains')
parser.add_argument('--device', type=str, help='Specify the device to use')
parser.add_argument('--epochs', type=int, help='No of epochs')
parser.add_argument('--replay', default=True, action=argparse.BooleanOptionalAction)
parser.add_argument('--em_frac', type=int, help='Percentage of the dataset to be used stored in memory for replay')
parser.add_argument('--lr', type=float, help='Learning rate')
parser.add_argument('--lr_decay', type=float, help='Learning rate decay factor for each dataset')
parser.add_argument('--epoch_decay', type=float, help='epochs will be decayed after training on each dataset')

parsed_args = parser.parse_args()
# order = parsed_args.order.split(',')
# print(f"Order : {order}, type : {type(order)}")
# print(f"Device : {parsed_args.device}, type : {type(parsed_args.device)}")
# print(f"Epochs : {parsed_args.epochs}, type : {type(parsed_args.epochs)}")
# print(f"Replay : {parsed_args.replay}, type : {type(parsed_args.replay)}")

print(parsed_args)
