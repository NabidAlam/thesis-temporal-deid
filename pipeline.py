import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, choices=['tsp_sam', 'samurai'], required=True)
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    if args.method == 'tsp_sam':
        from temporal.tsp_sam_runner import run_tsp_sam
        run_tsp_sam(args.input, args.output, args.config)

    elif args.method == 'samurai':
        from temporal.samurai_runner import run_samurai
        run_samurai(args.input, args.output, args.config)

if __name__ == '__main__':
    main()
