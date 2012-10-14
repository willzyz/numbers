function result = check_overlap(image, h, w, posh, posw)
    % function: tests whether cur block over-laps with any pre-layed blocks
    % rejects if overlaps or within 1 pixel
    
    [H, W] = size(image);
    minh = max(1, posh-1);
    maxh = min(H, posh+h);
    minw = max(1, posw-1);
    maxw = min(W, posw+w);
    
    block = image(minh:maxh, minw:maxw);
    if sum(block(:)) == 0
        result = false;
    else
        result = true;
    end
    
