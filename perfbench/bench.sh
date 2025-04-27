# Parse arguments
for arg in "$@"; do
  case $arg in
    --run_mode=NSYS)
      MODE="NSYS"
      shift
      ;;
    --run_mode=NCU)
      MODE="NCU"
      shift
      ;;
    *)
      echo "Invalid argument: $arg"
      exit 1
      ;;
  esac
done

if [[ "$MODE" == "NSYS" ]]; then
  echo "Running with NSYS..."
  nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu -o test -f true --cudabacktrace=true --osrt-threshold=10000 -x true python3 bench.py --run_mode NSYS
elif [[ "$MODE" == "NCU" ]]; then
  echo "Running with NCU..."
  ncu --set full -o test -f python3 bench.py --run_mode NCU
else
  echo "Error: No valid run_mode provided (use --run_mode=NSYS or --run_mode=NCU)."
  exit 1
fi