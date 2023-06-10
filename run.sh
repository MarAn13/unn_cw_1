#!/bin/bash

# pass params: 
#   data_type (individual, video, internet) 
#   mask_path ("" or path) 
#   vocab_tree ("" or path)
#   use_gpu (0 or 1)
function sfm_pipeline () {
    if [ "$1" == "internet" ]
    then
        local shared_camera=0
    else
        local shared_camera=1
    fi
    $colmap_dir feature_extractor \
    --database_path "$PROJECT/database.db" \
    --image_path "$DATA_ROOT/images" \
    --ImageReader.single_camera "$shared_camera" \
    --ImageReader.mask_path "$2" \
    --SiftExtraction.use_gpu "$4"
    if [ ! "$1" == "video" ]
    then
        if [ -z "$3" ]
        then
            $colmap_dir exhaustive_matcher \
            --database_path "$PROJECT/database.db" \
            --SiftMatching.use_gpu "$4"
        else
            $colmap_dir vocab_tree_matcher \
            --database_path "$PROJECT/database.db" \
            --VocabTreeMatching.vocab_tree_path "$3" \
            --SiftMatching.use_gpu "$4"
        fi
    else
        $colmap_dir sequential_matcher \
        --database_path "$PROJECT/database.db" \
        --SequentialMatching.vocab_tree_path "$3" \
        --SiftMatching.use_gpu "$4"
    fi
    if [ ! -d "$PROJECT/sparse" ]
    then
        mkdir "$PROJECT/sparse"
    fi
    # setup mapper options (with respect to data_type)
    local init_min_tri_angle=16
    local ba_global_images_ratio=1.1
    local ba_global_points_ratio=1.1
    local min_focal_length_ratio=0.1
    local max_focal_length_ratio=10
    local max_extra_param=1
    if [ "$1" == "video" ]
    then
        init_min_tri_angle=8
        ba_global_images_ratio=1.4
        ba_global_points_ratio=1.4
        max_extra_param=$((2**63-1))
    elif [ "$1" == "individual" ]
    then
        max_extra_param=$((2**63-1))
    fi
    $colmap_dir mapper \
    --database_path "$PROJECT/database.db" \
    --image_path "$DATA_ROOT/images" \
    --output_path "$PROJECT/sparse" \
    --Mapper.init_min_tri_angle $init_min_tri_angle \
    --Mapper.ba_global_images_ratio $ba_global_images_ratio \
    --Mapper.ba_global_points_ratio $ba_global_points_ratio \
    --Mapper.min_focal_length_ratio $min_focal_length_ratio \
    --Mapper.max_focal_length_ratio $max_focal_length_ratio \
    --Mapper.max_extra_param $max_extra_param

    if [ ! -d "$PROJECT/dense" ]
    then
        mkdir "$PROJECT/dense"
    fi
    $colmap_dir image_undistorter \
    --image_path "$DATA_ROOT/images" \
    --input_path "$PROJECT/sparse/0" \
    --output_path "$PROJECT/dense" \
    --output_type COLMAP
    $colmap_dir model_converter \
    --input_path "$PROJECT/dense/sparse" \
    --output_path "$PROJECT/dense/sparse"  \
    --output_type TXT
    return 0
}


# pass params: 
#   use_gpu (-1 or -2 for GPU and CPU respectively)
function mvs_pipeline () {
    $openmvs_dir/InterfaceCOLMAP \
    --working-folder "$DATA_ROOT/$PROJECT/dense" \
    --input-file "$DATA_ROOT/$PROJECT/dense" \
    --output-file "$DATA_ROOT/$PROJECT/mvs/model_colmap.mvs"
    #--cuda-device $1
    $openmvs_dir/DensifyPointCloud \
    --working-folder "$DATA_ROOT/$PROJECT/mvs" \
    --input-file "$DATA_ROOT/$PROJECT/mvs/model_colmap.mvs" \
    --output-file "$DATA_ROOT/$PROJECT/mvs/model_dense.mvs" \
    --verbosity 4
    #--cuda-device $1
    if [ "$1" == "no-mesh" ]
    then
        return 0
    fi
    $openmvs_dir/ReconstructMesh \
    --input-file "$DATA_ROOT/$PROJECT/mvs/model_dense.mvs" \
    --working-folder "$DATA_ROOT/$PROJECT/mvs/" \
    --output-file "$DATA_ROOT/$PROJECT/mvs/model_dense_mesh.mvs"
    #--cuda-device $1
    $openmvs_dir/RefineMesh \
    --working-folder "$DATA_ROOT/$PROJECT/mvs/" \
    --input-file "$DATA_ROOT/$PROJECT/mvs/model_dense_mesh.mvs" \
    --output-file "$DATA_ROOT/$PROJECT/mvs/model_dense_mesh_refine.mvs" \
    --resolution-level 1
    #--cuda-device $1
    $openmvs_dir/TextureMesh \
    --working-folder "$DATA_ROOT/$PROJECT/mvs/" \
    --input-file "$DATA_ROOT/$PROJECT/mvs/model_dense_mesh_refine.mvs" \
    --output-file "$DATA_ROOT/$PROJECT/mvs/model.obj" \
    --export-type obj
    #--cuda-device $1
    return 0
}


while getopts c:o:d: flag
do
    case "${flag}" in
        c) colmap_dir=${OPTARG};;
        o) openmvs_dir=${OPTARG};;
        d) workspace_dir=${OPTARG};;
        *) echo "incorrect flag" && exit 1;;
    esac
done
if [ -z "$colmap_dir" ] || [ -z "$openmvs_dir" ] || [ -z "$workspace_dir" ]
then
      echo "Error! Specify COLMAP and OpenMVS bin paths with -c and -o flags respectively. 
      Also don't forget to provide workspace directory with corresponding flag -d"
      exit 1
fi
vocab_trees=("" "$workspace_dir/vocab_tree_flickr100K_words32K.bin")
data_types=("individual" "video" "internet")
masks=("" "$DATA_ROOT/masks")
WHITE="\033[1;37m"
GREEN="\033[0;32m"
YELLOW="\033[1;33m"
RED="\033[0;31m"
file_log_time=$PWD/"time_log.txt"
echo "$PWD"
for dir in "$workspace_dir"/*
do
    if [ -d "$dir" ]; then
        dir=${dir//[.]}
        echo "DIR: $dir"
        cd "$dir" || (echo -e "$RED unable to cd to $dir" && exit 1)
        for data_type in "${data_types[@]}"
        do
            for mask in "${masks[@]}"
            do
                for vocab_tree in "${vocab_trees[@]}"
                do
                    # some checkers
                    if [[ -z "$vocab_tree" ]] 
                    then
                        vocab_tree_status=None
                    else
                        vocab_tree_status=On
                    fi
                    if [[ -z "$mask" ]] 
                    then
                        mask_status=None
                    else
                        if [[ "$mask" == "masks/object" ]] 
                        then
                            mask_status=Object
                        else
                            mask_status=Segment
                        fi
                    fi
                    # main code
                    PROJECT="project_${data_type}_${mask_status}_${vocab_tree_status}"
                    DATA_ROOT=$PWD
                    mask_path="$PWD/$mask"
                    echo -e "$YELLOW Reconstruction started"
                    echo -e "$WHITE"
                    echo "$DATA_ROOT"
                    if [ ! -d "$PROJECT/mvs" ]
                    then
                        mkdir "$PROJECT"
                    else
                        echo -e "$RED $PROJECT directory already exists" && exit 1
                    fi
                    start=$(date +%s)
                    sfm_pipeline "$data_type" "$mask" "$vocab_tree" 0
                    end_1=$(date +%s)
                    if [ ! -d "$PROJECT/mvs" ]
                    then
                        mkdir "$PROJECT/mvs"
                    fi
                    cd "$PROJECT/mvs" || (echo -e "$RED unable to cd to mvs/" && exit 1)
                    mvs_pipeline "mesh" #"no-mesh"
                    cd "../.." || (echo -e "$RED unable to cd to .." && exit 1)
                    end_2=$(date +%s)
                    # process info
                    runtime_1=$((end_1-start))
                    runtime_2=$((end_2-end_1))
                    runtime_3=$((end_2-start))
                    echo -e "$YELLOW Data type: $data_type, mask: $mask_status, vocab tree: $vocab_tree_status"
                    echo -e "$YELLOW      Elapsed SfM Time: $runtime_1 seconds" 
                    echo -e "$YELLOW      Elapsed MVS Time: $runtime_2 seconds"
                    echo -e "$YELLOW      Elapsed Total Time: $runtime_3 seconds"
                    echo -e "$WHITE"
                    echo "$PWD"
                    echo "Data type: $data_type, mask: $mask_status, vocab tree: $vocab_tree_status, SfM time: $runtime_1, MVS time: $runtime_2, total time: $runtime_3" >> "$file_log_time"
                    #rm -rf "$PROJECT/dense/images"
                done
            done
        done
        cd "../.." || (echo -e "$RED unable to cd to ../.." && exit 1)
    fi
done
echo -e "$GREEN SUCCESS"