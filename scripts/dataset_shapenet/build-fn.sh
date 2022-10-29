source dataset_shapenet/config.sh
# Make output directories
mkdir -p $BUILD_PATH

# Run build
for c in ${CLASSES[@]}; do
  echo "Processing class $c"
  input_path_c=$INPUT_PATH/$c
  build_path_c=$BUILD_PATH/$c

  mkdir -p $build_path_c/4_fn

  echo "Process watertight meshes"
  python sample_mesh-fn.py $build_path_c/4_watertight_scaled \
      --n_proc 8\
      --pointing_folder $build_path_c/4_fn

    
done
