import argparse
import numpy as np
import model as M
import chess

def calculate_statistics(weights, x0, x1, y0, y1):
    """
    Calculates L1, L2, and L_inf norms for vectors in the specified quadrant.
    Assumes X represents columns (embedding dimensions) and Y represents rows (items).
    """
    max_y, max_x = weights.shape
    
    # Ensure bounds are within the matrix and properly ordered
    x_start, x_end = max(0, min(x0, x1)), min(max_x - 1, max(x0, x1))
    y_start, y_end = max(0, min(y0, y1)), min(max_y - 1, max(y0, y1))

    # Inclusive slicing
    sub_matrix = weights[y_start:y_end+1, x_start:x_end+1]
    
    if sub_matrix.size == 0:
        print(f"Region (x: {x_start}-{x_end}, y: {y_start}-{y_end}) is empty.")
        return

    # Calculate norms along the rows (axis=1)
    l1_norms = np.linalg.norm(sub_matrix, ord=1, axis=1)
    l2_norms = np.linalg.norm(sub_matrix, ord=2, axis=1)
    linf_norms = np.linalg.norm(sub_matrix, ord=np.inf, axis=1)
    
    l1_norms_t = np.linalg.norm(sub_matrix, ord=1, axis=0)
    l2_norms_t = np.linalg.norm(sub_matrix, ord=2, axis=0)
    linf_norms_t = np.linalg.norm(sub_matrix, ord=np.inf, axis=0)
    
    percentiles = [50, 90, 95, 99, 99.9, 99.99]
    
    print(f"\n--- Statistics for Region (X: {x_start} to {x_end}, Y: {y_start} to {y_end}) ---")
    print(f"Extracted Shape: {sub_matrix.shape}")
    
    stats_to_print = [
        ("L1", l1_norms),
        ("L2", l2_norms),
        ("L_inf", linf_norms),
        ("L1_T", l1_norms_t),
        ("L2_T", l2_norms_t),
        ("L_inf_T", linf_norms_t)
    ]
    
    for name, norms in stats_to_print:
        p_values = np.percentile(norms, percentiles)
        print(f"\n{name} Norm Percentiles:")
        for p, val in zip(percentiles, p_values):
            print(f"  {p:>5}th: {val:.6f}")
    print("-" * 65)


def parse_region(region_str):
    """Parses a string 'x0,x1,y0,y1' into a tuple of ints."""
    try:
        parts = region_str.replace(',', ' ').split()
        if len(parts) != 4:
            raise ValueError
        return tuple(map(int, parts))
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid region format: '{region_str}'. Expected format: x0,x1,y0,y1"
        )


def interactive_mode(weights):
    max_y, max_x = weights.shape
    print(f"\n[Interactive Mode] Loaded weight matrix of shape (Y={max_y}, X={max_x}).")
    print("Enter regions in the format: x0 x1 y0 y1 (inclusive bounds).")
    print("Type 'q' or 'quit' to exit.")
    
    while True:
        user_input = input("\nRegion (x0 x1 y0 y1): ").strip()
        if user_input.lower() in ['q', 'quit', 'exit']:
            print("Exiting interactive mode.")
            break
            
        try:
            # Allow both spaces and commas
            parts = user_input.replace(',', ' ').split()
            if len(parts) != 4:
                print("Error: Please enter exactly four integers. Type 'q' to quit.")
                continue
                
            x0, x1, y0, y1 = map(int, parts)
            calculate_statistics(weights, x0, x1, y0, y1)
            
        except ValueError:
            print("Error: Invalid input. Ensure you are using integers.")


def main():
    parser = argparse.ArgumentParser(description="Analyzes input weight embeddings (L1, L2, L_inf norms).")
    parser.add_argument("model", help="Source model (can be .ckpt, .pt or .nnue)")
    parser.add_argument(
        "--regions", 
        nargs="*", 
        type=parse_region, 
        help="List of quadrants to evaluate in the format x0,x1,y0,y1 (e.g., --regions 0,63,0,1023)"
    )
    parser.add_argument(
        "-i", "--interactive", 
        action="store_true", 
        help="Run in interactive mode to query regions on the fly."
    )
    parser.add_argument(
        "-p", "--piece", 
        type=chess.Piece.from_symbol,
        default=None,
        help="filter for pieces. Example: --piece N for knights. Only works with Full_Threats features."
    )
    parser.add_argument("--l1", type=int, default=M.ModelConfig().L1)
    M.add_feature_args(parser)
    
    args = parser.parse_args()

    supported_features = ("HalfKAv2_hm", "HalfKAv2_hm^", "Full_Threats")
    assert args.features in supported_features, f"Features must be one of {supported_features}"
    feature_set = M.get_feature_set_from_name(args.features)

    print(f"Loading model: {args.model}")
    model = M.load_model(
        args.model, feature_set, M.ModelConfig(L1=args.l1), M.QuantizationConfig()
    )

    # Extract the input weights (embeddings)
    weights = M.coalesce_ft_weights(model.feature_set, model.input).numpy()
    if args.piece:
        idxs = feature_set.features[0].get_indeces(p=args.piece)
        weights = weights[idxs]

    # Default: Analyse the whole matrix if no specific regions or interactive mode is set
    if not args.regions and not args.interactive:
        max_y, max_x = weights.shape
        print("No regions specified. Analyzing the entire matrix by default.")
        calculate_statistics(weights, 0, max_x - 1, 0, max_y - 1)
    
    # Process CLI regions if provided
    if args.regions:
        for region in args.regions:
            calculate_statistics(weights, *region)
            
    # Enter interactive mode if flagged
    if args.interactive:
        interactive_mode(weights)


if __name__ == "__main__":
    main()