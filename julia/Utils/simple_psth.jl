function simple_PSTH(spk_times,event_times,bin_size, start, stop)
        #inputs: - spk_times in sec
        # - event
        c = [histcounts(spk_times,floor.(event_times[i]+start,digits=2):bin_size:ceil(event_times[i]+stop,digits=2)) for i in 1:length(event_times)]
        spk_cnt = sum(c,dims=1)
        return spk_cnt./length(event_times)./bin_size 
        end