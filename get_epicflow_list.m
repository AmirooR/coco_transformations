function get_epicflow_list(img1_path_list, img2_path_list, flow_path_list)
    parfor i=1:length(img1_path_list)
        disp(flow_path_list{i});
        img1 = imread(img1_path_list{i});
        img2 = imread(img2_path_list{i});
        get_epicflow(img1, img2, flow_path_list{i}, i);
    end
