delete 
    from datapoints 
    where classification = 'SubdivisionAction.UNDEFINED' 
    or (x = 0 and y = 0 and sx = 1 and sy = 1)
;
