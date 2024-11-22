function simple_raster_units(spk_times, event_time, start, stop)
    n_units = size(spk_times, 1)
    X = []
    Y = []
    for u in 1:n_units
    ts_in_window = find_spks_in_window(spk_times[u], event_time, start, stop)
    units_rows = ones(length(ts_in_window)).*u
    append!(X,ts_in_window)
    append!(Y,units_rows)
    end
    return X,Y
end

function find_spks_in_window(spk_times_unit, event_time, start, stop)
    #inputs: - spk_times in sec
    # - event_times in sec
    #bin_size in sec
    #start in sec, for the time before the event_times
    #stop in sec for the time after the event_times
    centered_spk_ts = spk_times_unit .- event_time
    index_in_window = intersect(findall(centered_spk_ts.>start),findall(centered_spk_ts.<stop))
    ts_in_window  = centered_spk_ts[index_in_window]
    end