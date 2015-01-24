function [K_test] = mklv2_make_test_kernel(X_text_dev, X_image_dev, weight, info_kernel, mkl_options, X_text_sv, X_image_sv, sigma)

if ~isequal(size(X_text_dev, 1), size(X_image_dev, 1))
    error('unmatched dev vector size');
end
if ~isequal(size(X_text_sv, 1), size(X_image_sv, 1))
    error('unmatched support vector size');
end

ind = find(sigma);
K_test = zeros(size(X_text_dev,1),size(X_text_sv,1));
for i=1:length(ind);
    layer_idx=ind(i); 
    keyboard;
    if layer_idx<=size(sigma, 2)/2
        Kr = svmkernel(X_text_dev(:,info_kernel(layer_idx).variable), info_kernel(layer_idx).kernel, ...
            info_kernel(layer_idx).kerneloption, X_text_sv(:,info_kernel(layer_idx).variable));
    else
        Kr = svmkernel(X_image_dev(:,info_kernel(layer_idx).variable), info_kernel(layer_idx).kernel, ...
            info_kernel(layer_idx).kerneloption, X_image_sv(:,info_kernel(layer_idx).variable));        
    end

    Kr = Kr * weight(layer_idx);
    K_test = K_test + Kr*sigma(layer_idx);
end;
