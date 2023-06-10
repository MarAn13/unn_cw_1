function scores = compare_point_clouds(path_to_dir, reference)
    % adding necessary libs
    addpath("point_fusion\", "attribute_estimation\", "structural_similarity\", "voxelization\");

    % get ply files
    p_clouds = get_point_clouds(path_to_dir);

    % setup reference to compare with
    if reference == "None"
        use_reference = false;
        reference = p_clouds;
    else
        use_reference = true;
    end

    % configure parameters
    OPTIONS = configure_params();

    TEST_PARAMS = ["GEOM" "CURV" "NORM" "COLOR"];

    for k=1:length(TEST_PARAMS)
        for i=1:length(p_clouds)
            for j=(i * not(use_reference))+1:length(string(reference))
                % load point clouds
                disp("P_CLOUD");
                disp(p_clouds(i));
                A = pcread(p_clouds(i));
                if use_reference
                    B = pcread(reference);
                else
                    B = pcread(reference(j));
                end

                % adjust type estimation parameters
                OPTIONS = adjust_params(OPTIONS, TEST_PARAMS(k));
                
                % configure fitting parameters
                OPTIONS = configure_fitting(OPTIONS);
    
                % estimate SSIM
                pssim = estimate_ssim(A, B, OPTIONS);
    
                % create score structure
                field = "iteration" + string(i) + string(j);
                field_name_A = split(p_clouds(i), "/");
                if use_reference
                    field_name_B = split(reference, "\");
                else
                    field_name_B = split(reference(j), "/");
                end
                disp("FIELDA");
                disp(field_name_A);
                disp("FIELDB");
                disp(field_name_B);
                field_name = string(field_name_A(end-2)) + "; " + string(field_name_B(end-2));
                
                % assign SSIM to structure
                score_log.(field).name = field_name;
                score_log.(field).score = pssim;
            end
        end
        % write all scores to file
        file = fopen("score_log_" + TEST_PARAMS(k) + ".json",'w');
        fprintf(file, "%s", jsonencode(score_log, "PrettyPrint", true));
        fclose(file);
    
        % write essential scores to file
        file = fopen("score_" + TEST_PARAMS(k) + ".json",'w');
        fn = fieldnames(score_log);
        disp(fn);
        for i=1:numel(fn)
            iter = score_log.(fn{i});
            fields = vertcat(fieldnames(iter.score));
            idx = contains(fields, "Sym");
            fields = fields(idx);
            iter_field = fn{i};
            for j=1:length(fields)
                scores.(iter_field).name = iter.name;
                scores.(iter_field).score.(fields{j}) = iter.score.(fields{j});
            end
        end
        fprintf(file, "%s", jsonencode(scores, "PrettyPrint", true));
        fclose(file);
    end
end

%% get ply files
function p_clouds = get_point_clouds(path_to_dir)
    pathDir = dir(path_to_dir);
    dirFolders = pathDir([pathDir(:).isdir]);
    dirFolders = dirFolders(~ismember({dirFolders(:).name},{'.','..'}));
    p_clouds = "";
    for i = 1:length(dirFolders)
      baseFolderName = dirFolders(i).name;
      if baseFolderName == "images" || baseFolderName == "masks"
          continue
      end
      fullFolderName = path_to_dir + "/" + baseFolderName;
      search_dir = fullFolderName + "/mvs";
      delimiter = ";";
      if isfolder(search_dir)
          files = dir(fullfile(search_dir,'*.ply'));
          files_length = size(files);
          if files_length(1) ~= 1
              error("check .ply files in " + search_dir);
          end
          if i == length(dirFolders)
              delimiter = "";
          end
          p_clouds = p_clouds + search_dir + "/" + files.name + delimiter;
      end
    end
    p_clouds = split(p_clouds, ";");
end

%% adjust parameters
function OPTIONS = adjust_params(OPTIONS, score_type)
    param_GEOM = false;
    param_NORM = false;
    param_CURV = false;
    param_COLOR = false;
    if score_type == "GEOM"
        param_GEOM = true;
    elseif score_type == "NORM" || score_type == "CURV"
        param_NORM = true;
        param_CURV = true;
    else
        param_COLOR = true;
    end
    OPTIONS.PARAMS.ATTRIBUTES.GEOM = param_GEOM;
    OPTIONS.PARAMS.ATTRIBUTES.NORM = param_NORM;
    OPTIONS.PARAMS.ATTRIBUTES.CURV = param_CURV;
    OPTIONS.PARAMS.ATTRIBUTES.COLOR = param_COLOR;
end

%% configure parameters
function OPTIONS = configure_params()
    PARAMS.ATTRIBUTES.GEOM = true;
    PARAMS.ATTRIBUTES.NORM = false;
    PARAMS.ATTRIBUTES.CURV = false;
    PARAMS.ATTRIBUTES.COLOR = false;
    
    PARAMS.ESTIMATOR_TYPE = {'VAR'};
    PARAMS.POOLING_TYPE = {'Mean'};
    PARAMS.NEIGHBORHOOD_SIZE = 12;
    PARAMS.CONST = eps(1);
    PARAMS.REF = 0;
    
    QUANT.VOXELIZATION = false;
    QUANT.TARGET_BIT_DEPTH = 9;
    
    OPTIONS.PARAMS = PARAMS;
    OPTIONS.QUANT = QUANT;
end

%% configure fitting parameters
function OPTIONS = configure_fitting(OPTIONS)
    FITTING.SEARCH_METHOD = 'knn';
    if strcmp(FITTING.SEARCH_METHOD, 'rs')
        FITTING.ratio = 0.01;
    elseif strcmp(FITTING.SEARCH_METHOD, 'knn')
        FITTING.knn = 12;
    end
    FITTING.SEARCH_SIZE = [];

    % normals and curvatures estimation
    if OPTIONS.PARAMS.ATTRIBUTES.NORM || OPTIONS.PARAMS.ATTRIBUTES.CURV
        if strcmp(FITTING.SEARCH_METHOD, 'rs')
            FITTING.SEARCH_SIZE = round(FITTING.ratio * double(max(max(A.Location) - min(A.Location))));
        else
            FITTING.SEARCH_SIZE = FITTING.knn;
        end
    end

    OPTIONS.FITTING = FITTING;
end

%% estimate point clouds SSIM
function [pssim] = estimate_ssim(A, B, OPTIONS)
    % sort geometry
    [geomA, idA] = sortrows(A.Location);
    if ~isempty(A.Color)
        colorA = A.Color(idA, :);
        A = pointCloud(geomA, 'Color', colorA);
    else
        A = pointCloud(geomA);
    end
    
    [geomB, idB] = sortrows(B.Location);
    if ~isempty(B.Color)
        colorB = B.Color(idB, :);
        B = pointCloud(geomB, 'Color', colorB);
    else
        B = pointCloud(geomB);
    end
    
    % point fusion
    A = pc_fuse_points(A);
    B = pc_fuse_points(B);

    % voxelization
    if OPTIONS.QUANT.VOXELIZATION
        A = pc_vox_scale(A, [], OPTIONS.QUANT.TARGET_BIT_DEPTH);
        B = pc_vox_scale(B, [], OPTIONS.QUANT.TARGET_BIT_DEPTH);
    end
    
    % normals and curvatures estimation
    if OPTIONS.PARAMS.ATTRIBUTES.NORM || OPTIONS.PARAMS.ATTRIBUTES.CURV
        if strcmp(OPTIONS.FITTING.SEARCH_METHOD, 'rs')
            OPTIONS.FITTING.SEARCH_SIZE = round(OPTIONS.FITTING.ratio * double(max(max(A.Location) - min(A.Location))));
        else
            OPTIONS.FITTING.SEARCH_SIZE = OPTIONS.FITTING.knn;
        end
        [normA, curvA] = pc_estimate_norm_curv_qfit(A, OPTIONS.FITTING.SEARCH_METHOD, OPTIONS.FITTING.SEARCH_SIZE);
        [normB, curvB] = pc_estimate_norm_curv_qfit(B, OPTIONS.FITTING.SEARCH_METHOD, OPTIONS.FITTING.SEARCH_SIZE);
    end
    
    % set custom structs with required fields
    sA.geom = A.Location;
    sB.geom = B.Location;
    if OPTIONS.PARAMS.ATTRIBUTES.NORM
        sA.norm = normA;
        sB.norm = normB; 
    end
    if OPTIONS.PARAMS.ATTRIBUTES.CURV
        sA.curv = curvA;
        sB.curv = curvB;
    end
    if OPTIONS.PARAMS.ATTRIBUTES.COLOR
        sA.color = A.Color;
        sB.color = B.Color;
    end
    
    % compute structural similarity scores
    [pssim] = pointssim(sA, sB, OPTIONS.PARAMS);
end