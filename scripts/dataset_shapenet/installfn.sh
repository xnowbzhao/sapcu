source dataset_shapenet/config.sh

# Function for processing a single model
reorganize() {
  modelname=$3
  output_path="$2/$modelname"
  build_path=$1

  points_file="$build_path/4_fn/$modelname.npz"
  points_out_file="$output_path/fn.npz"

  echo "Copying model $output_path"

  cp $points_file $points_out_file
}

export -f reorganize

# Make output directories
mkdir -p $OUTPUT_PATH

# Run build
for c in ${CLASSES[@]}; do
  echo "Parsing class $c"
  BUILD_PATH_C=$BUILD_PATH/$c
  OUTPUT_PATH_C=$OUTPUT_PATH/$c
  mkdir -p $OUTPUT_PATH_C
  echo $BUILD_PATH_C
  ls $BUILD_PATH_C/4_fn/ | sed -e 's/\.npz$//' | parallel -P $NPROC --timeout $TIMEOUT \
    reorganize $BUILD_PATH_C $OUTPUT_PATH_C {}

done
