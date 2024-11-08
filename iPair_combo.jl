using CSV # for manipulating CSV files
using StatsBase # countmap for histograms
using Distributed # distributing for
using DataStructures # PriorityQueue for keeping information gains of partitions in order
using IterTools # subsets for generating combinations of partitions
using Base.Iterators # product for generating products of feature values
using Profile # profiling

const MI_BASE = 2 # mutual information base
const MI_MERGE_THRESH = 0.5 # mutual information threshold to merge
const SEC_MERGE_FAIL_PROP = 0.05 # threshold to cut merges
const SEC_COMP_FAILS = 10 # threshold to cut compresses
const MIN_VAL_FREQ = 0 # minimum feature value frequency to consider to compress
const SUB_SAMPLE_DIVISOR = 1 # how many subsamples to keep

function main()

    if nprocs() < Sys.CPU_THREADS
        # Distributed.addprocs(Sys.CPU_THREADS - nprocs())
        Distributed.addprocs(4)
    end

    @everywhere using LinearAlgebra # dot for getting sumproduct of two array of bits

    @everywhere mutable struct CodeDictEntry # code dictionary entry
        freq::Int64 # keep sum of code for efficiency
        code::BitArray # corresponding code in data
    end

    @everywhere mutable struct DataDictEntry # data dictionary entry
        entropy::Float64 # keep entropy of fields for efficiency
        codes::Array{BitArray} # all codes related to the fields
    end

    @everywhere function ent(f, base=MI_BASE)::Float64
        # if !iszero(x)
        #     return -x / n * log(base, x / n)
        # else
        #     return 0.0
        # end
        ifelse(f > 0.0, -f * log(base, f), 0.0)
    end

    @everywhere function overlappedEnt(codes_a, codes_b, ent_base)
        total_ent = 0.0
        # for code_a in codes_a, code_b in codes_b
            # overlapped_freq = dot(code_a, code_b)
            # if overlapped_freq > 0
            #     total_ent += ent(overlapped_freq / n)
            # end
            # total_ent += ent(dot(code_a, code_b) / ent_base)
        # end
        # return sum(ent(dot(code_a, code_b) / ent_base)::Float64 for code_a in codes_a, code_b in codes_b)
        for code_a in codes_a, code_b in codes_b
            total_count = LinearAlgebra.dot(code_a, code_b)
            total_ent += ent(total_count / ent_base)
        end
        return total_ent
    end

    data_path = "."
    data_file = "synthesised_10k_ano_1m_total_200features_5real.csv"

    inc_records = 1000000 # 40000
    data_raw = first(CSV.read(string(data_path, data_file)), inc_records) # string( , ) to concatenate two strings
    delete!(data_raw, :Column1) # since the first column in the data file is record index
    # select!(data_raw, (:Column1))
    delete!(data_raw, :CLASS) # since the class column in the data file is correct label
    # select!(data_raw, !(:CLASS))
    n, m = size(data_raw) # n is number of records, m is number of fields
    @everywhere n, m = $n, $m
    @everywhere MI_BASE = $MI_BASE
    println("number of records ", n, ", number of fields ", m)

    @time begin
    partition_code_dict = Dict{Array{Symbol, 1}, Dict{Array{String, 1}, CodeDictEntry}}() # ((F0), (F1)) -> (F0_0, F1_0) -> (count, cost, code)
    data_raw_code_dict = Dict{Array{Symbol, 1}, DataDictEntry}() # ((F0), (F1)) -> (cost, array of codes)
    subsample_indices = StatsBase.sample(1:n, n ÷ SUB_SAMPLE_DIVISOR)
    for field::Symbol in names(data_raw) # loop through all fields
        partition_code_dict[[field]] = Dict{Array{String, 1}, CodeDictEntry}()
        data_raw_code_dict[[field]] = DataDictEntry(0.0, Array{BitArray, 1}()) # DataDictEntry(entropy, codes)
        value_freq_dict::Dict{String, Int64} = StatsBase.countmap(data_raw[!, field]) # F0_1 -> 2; F0_2 -> 4, ...
        #= sum_ent = 0.0
        for (value::String, freq::Int64) in value_freq_dict
            partition_code_dict[[field]][[value]] = CodeDictEntry(freq, data_raw[!, field] .== value)
            sum_ent += ent(freq / n)
            push!(data_raw_code_dict[[field]].codes, partition_code_dict[[field]][[value]].code)
        end =#
        for (value::String, freq::Int64) in value_freq_dict
            partition_code_dict[[field]][[value]] = CodeDictEntry(freq, data_raw[!, field] .== value)
        end
        value_freq_dict = StatsBase.countmap(data_raw[subsample_indices, field])
        sum_ent = 0.0
        for (value::String, freq::Int64) in value_freq_dict
            push!(data_raw_code_dict[[field]].codes, data_raw[subsample_indices, field] .== value)
            sum_ent += ent(freq / (n ÷ SUB_SAMPLE_DIVISOR))
        end
        data_raw_code_dict[[field]].entropy = sum_ent
        println("code dict and data dict on ", field, " finished")
    end

    # free memory
    data_raw_names = names(data_raw)
    # data_raw = nothing
    # GC.gc()

    # define dictionaries to make (part_a, part_b) loops parallel
    @everywhere rd_pair(part_a::Tuple{Array{Symbol, 1}, Array{Symbol, 1}, Array{BitArray, 1}, Array{BitArray, 1}, Int64}, part_b::Tuple{Array{Symbol, 1}, Array{Symbol, 1}, Array{BitArray, 1}, Array{BitArray, 1}, Int64}) = rd_pair(rd_pair(Dict{Tuple{Array{Symbol, 1}, Array{Symbol, 1}}, Float64}(), part_a), part_b)
    @everywhere rd_pair(dict::Dict{Tuple{Array{Symbol, 1}, Array{Symbol, 1}}, Float64}, part::Tuple{Array{Symbol, 1}, Array{Symbol, 1}, Array{BitArray, 1}, Array{BitArray, 1}, Int64}) =
        begin
            dict[(part[1], part[2])] = overlappedEnt(part[3], part[4], part[5]);
            dict
        end
    @everywhere rd_pair(dict_a::Dict{Tuple{Array{Symbol, 1}, Array{Symbol, 1}}, Float64}, dict_b::Dict{Tuple{Array{Symbol, 1}, Array{Symbol, 1}}, Float64}) = merge!(dict_a, dict_b)

    # define dictionaries to make (code_a, code_b) loops parallel
    @everywhere rd_code_pair(part_a::Tuple{Array{Symbol, 1}, Array{Symbol, 1}, BitArray, BitArray, Int64}, part_b::Tuple{Array{Symbol, 1}, Array{Symbol, 1}, BitArray, BitArray, Int64}) = rd_code_pair(rd_code_pair(Dict{Tuple{Array{Symbol, 1}, Array{Symbol, 1}}, Float64}(), part_a), part_b)
    @everywhere rd_code_pair(dict::Dict{Tuple{Array{Symbol, 1}, Array{Symbol, 1}}, Float64}, part::Tuple{Array{Symbol, 1}, Array{Symbol, 1}, BitArray, BitArray, Int64}) =
        begin
            total_count = LinearAlgebra.dot(part[3], part[4])
            dict[(part[1], part[2])] = get(dict, (part[1], part[2]), 0) + ent(total_count / part[5]);
            dict
        end
    @everywhere rd_code_pair(dict_a::Dict{Tuple{Array{Symbol, 1}, Array{Symbol, 1}}, Float64}, dict_b::Dict{Tuple{Array{Symbol, 1}, Array{Symbol, 1}}, Float64}) = merge!(+, dict_a, dict_b)

    # define arrays to make (part_a, part_b) loops parallel
    @everywhere ra_pair(part_a::Tuple, part_b::Tuple) = ra_pair(ra_pair(Vector(), part_a), part_b)
    @everywhere ra_pair(arr::Vector, part::Tuple) =
        begin push!(arr,
            ((part[1], part[2]), overlappedEnt(part[3], part[4], part[5])));
            arr
        end
    @everywhere ra_pair(arr_a::Vector, arr_b::Vector) = begin arr_a = vcat(arr_a, arr_b) end

    partition_ig_pq = DataStructures.PriorityQueue{Tuple{Array{Symbol, 1}, Array{Symbol, 1}}, Float64}()
    to_distribute = []
    for (partition_a::Symbol, partition_b::Symbol) in IterTools.subsets(data_raw_names, 2)
        for code_a in data_raw_code_dict[[partition_a]].codes, code_b in data_raw_code_dict[[partition_b]].codes
            push!(to_distribute, ([partition_a], [partition_b], code_a, code_b))
        end
    end
    overlapped_ent_to_add = @distributed (rd_code_pair) for comp in to_distribute
        (comp[1], comp[2], comp[3], comp[4], n ÷ SUB_SAMPLE_DIVISOR)
    end
    for (part::Tuple{Array{Symbol, 1}, Array{Symbol, 1}}, overlapped_ent::Float64) in overlapped_ent_to_add # parallel dict
        ig::Float64 = (data_raw_code_dict[part[1]].entropy + data_raw_code_dict[part[2]].entropy - overlapped_ent) / 2
        enqueue!(partition_ig_pq, (part[1], part[2]), -ig) # min heap so take most negative
        println("pushed ", (part[1], part[2]), ' ', -ig, " entry to priority queue for merging")
    end

    failsAfterSuccMerge = 0
    secMergeFailThresh = trunc(Int, length(partition_ig_pq) * SEC_MERGE_FAIL_PROP) + 1
    current_partition_step = 0 # keep track of number of steps taken in partition loop
    while length(partition_ig_pq) > 0 && failsAfterSuccMerge < secMergeFailThresh
        (partition_pair::Tuple{Array{Symbol, 1}, Array{Symbol, 1}}, current_ig::Float64) = DataStructures.peek(partition_ig_pq)
        dequeue!(partition_ig_pq)
        @everywhere current_partition_pair = $partition_pair
        println("current partition pair ", partition_pair, ' ', current_ig)
        # total_cost = current total cost
        # singleton_freq_dict = singleton frequency dictionary to update
        # total_freq = total count of code
        # singleton_cost = singletone cost
        # total_cost, singleton_freq_dict, total_freq, singleton_cost = encoding_cost(partition_code_dict)
@time begin
        encoded_data_cost = 0.0
        pattern_cost = 0.0 # think
        freqs = Array{Int64, 1}(); # array of freqency in all partitions and all feature values
        for (part, code_dict) in partition_code_dict
            for (value, code_entry) in code_dict
                push!(freqs, code_entry.freq)
            end
        end
        total_freq::Int64 = sum(freqs)
        pattern_cost -= sum(log.(MI_BASE, freqs / total_freq))
        encoded_data_cost -= sum(freqs .* log.(MI_BASE, freqs / total_freq))
        singleton_freq_dict = Dict{String, Int64}() # think
        for (part, code_dict) in partition_code_dict
            for value in keys(code_dict)
                # merge!(+, singleton_freq_dict, countmap(flattenFeatureValues(value)))
                merge!(+, singleton_freq_dict, countmap(value))
            end
        end
        singleton_freqs::Array{Int64, 1} = collect(values(singleton_freq_dict))
        total_singleton_freq::Int64 = sum(singleton_freqs)
        singleton_cost::Float64 = -sum(singleton_freqs .* log.(MI_BASE, singleton_freqs / total_singleton_freq))
        total_cost::Float64 = encoded_data_cost + singleton_cost + pattern_cost
        global current_partition_step += 1
        println("current partition step ", current_partition_step, ' ', total_cost)
        # initialize current code table, need to sort by (length, cost)
        current_partition_pair_code_dict = deepcopy(merge(partition_code_dict[current_partition_pair[1]], partition_code_dict[current_partition_pair[2]]))
        current_partition_pair_code_dict_best = deepcopy(current_partition_pair_code_dict)
        overlapped_code = BitArray(zeros(n))
        overlapped_freq = 0
        value_a_freq = 0
        value_b_freq = 0
        unique_value_pq = PriorityQueue{Tuple{Array{String, 1}, Array{String, 1}}, Int64}()
        for elem_a in partition_code_dict[current_partition_pair[1]], elem_b in partition_code_dict[current_partition_pair[2]]
            overlapped_freq = dot(elem_a[2].code, elem_b[2].code)
            if overlapped_freq > 0
                enqueue!(unique_value_pq, (elem_a[1], elem_b[1]), -overlapped_freq)
            end
        end
        total_cost_updated = total_cost
        mean_reduced_cost = Array{Float64, 1}() # keep track of the threshold to determine whether to merge
        failsAfterSuccComp = 0
        while length(unique_value_pq) > 0 && failsAfterSuccComp < SEC_COMP_FAILS
            value_pair = dequeue!(unique_value_pq)
            if haskey(current_partition_pair_code_dict, value_pair[1]) && haskey(current_partition_pair_code_dict, value_pair[2])
                overlapped_code = current_partition_pair_code_dict[value_pair[1]].code .& current_partition_pair_code_dict[value_pair[2]].code
                overlapped_freq = mapreduce(count_ones, +, overlapped_code.chunks)
                value_a_freq = current_partition_pair_code_dict[value_pair[1]].freq
                value_b_freq = current_partition_pair_code_dict[value_pair[2]].freq
            else
                println("key not exist ", value_pair[1], ' ', value_pair[2])
                overlapped_code = BitArray(zeros(n))
                overlapped_freq = 0
                value_a_freq = 0
                value_b_freq = 0
            end
            if overlapped_freq > MIN_VAL_FREQ # 0
                # available upon request
            end
        end
        current_partition_pair_length::Int64 = length(current_partition_pair[1]) + length(current_partition_pair[2])
        mi::Float64 = current_partition_pair_length * current_ig
end
        if length(mean_reduced_cost) > 0 && sum(mean_reduced_cost) / length(mean_reduced_cost) > -1 / mi
            global failsAfterSuccMerge = 0
            # sort dictionary according to (length, entropy)
            # current_partition_pair_code_dict_best = Dict(sort(collect(current_partition_pair_code_dict_best), by = x -> (length(flattenFeatureValues(x[1])), x[2]), rev = true))
            # partition_code_dict[current_partition_pair] = copy(current_partition_pair_code_dict_best)
            partition_list = vcat(current_partition_pair[1], current_partition_pair[2])
            partition_code_dict[partition_list] = current_partition_pair_code_dict_best
            delete!(partition_code_dict, current_partition_pair[1])
            delete!(partition_code_dict, current_partition_pair[2])
            # remove items in partition_ig_pq
            # not working since changing order filter!(elem -> !(elem[1] in current_partition_pair) && !(elem[2] in current_partition_pair), partition_ig_pq)
            for (part::Tuple{Array{Symbol, 1}, Array{Symbol, 1}}, ig::Float64) in partition_ig_pq
                if (part[1] ⊆ current_partition_pair[1]) || (part[1] ⊆ current_partition_pair[2]) || (part[2] ⊆ current_partition_pair[1]) || (part[2] ⊆ current_partition_pair[2])
                    delete!(partition_ig_pq, part)
                end
            end
            # insert current_partition_pair into data_raw_code_dict
@time begin
            data_raw_code_dict[partition_list] = DataDictEntry(0.0, Array{BitArray, 1}())
            total_ent = 0.0
            for code_a in data_raw_code_dict[current_partition_pair[1]].codes
                for code_b in data_raw_code_dict[current_partition_pair[2]].codes
                    overlapped_code::BitArray = code_a .& code_b
                    overlapped_freq = mapreduce(count_ones, +, overlapped_code.chunks)
                    if overlapped_freq > 0
                        push!(data_raw_code_dict[partition_list].codes, overlapped_code)
                        total_ent += ent(overlapped_freq / (n ÷ SUB_SAMPLE_DIVISOR))
                    end
                end
            end
            data_raw_code_dict[partition_list].entropy = total_ent
            # @everywhere data_raw_code_dict = $data_raw_code_dict
end
@time begin
            # all_other_parts::Array{Array{Symbol, 1}, 1} = filter(elem -> elem != partition_list, collect(keys(partition_code_dict))) # map(elem -> (current_partition_pair, elem), filter(elem -> elem != current_partition_pair, collect(keys(partition_code_dict))))
            all_other_parts::Array{Array{Symbol, 1}, 1} = []
            for part in keys(partition_code_dict)
                if part != partition_list
                    mi_avg_intra_current = 0
                    for (x, y) in IterTools.subsets(partition_list, 2)
                        mi_avg_intra_current += data_raw_code_dict[[x]].entropy + data_raw_code_dict[[y]].entropy - overlappedEnt(data_raw_code_dict[[x]].codes, data_raw_code_dict[[y]].codes, n ÷ SUB_SAMPLE_DIVISOR)
                    end
                    mi_avg_inter = 0
                    for (x, y) in IterTools.product(partition_list, part)
                        mi_avg_inter += data_raw_code_dict[[x]].entropy + data_raw_code_dict[[y]].entropy - overlappedEnt(data_raw_code_dict[[x]].codes, data_raw_code_dict[[y]].codes, n ÷ SUB_SAMPLE_DIVISOR)
                    end
                    if MI_MERGE_THRESH * mi_avg_intra_current / (current_partition_pair_length * (current_partition_pair_length - 1) / 2) < mi_avg_inter / (length(partition_list) * length(part)) * current_partition_pair_length * length(part)
                        push!(all_other_parts, part)
                    end
                end
            end
            overlapped_ent_to_add = Dict()
            if length(all_other_parts) <= 0
            elseif length(all_other_parts) == 1
                overlapped_ent_to_add[(partition_list, all_other_parts[1])] = overlappedEnt(data_raw_code_dict[partition_list].codes, data_raw_code_dict[all_other_parts[1]].codes, n ÷ SUB_SAMPLE_DIVISOR)
            else
                overlapped_ent_to_add::Dict{Tuple{Array{Symbol, 1}, Array{Symbol, 1}}, Float64} = @distributed (rd_pair) for part in all_other_parts
                # overlapped_ent_to_add = @distributed (ra_pair) for part in all_other_parts
                # overlapped_ent_to_add::Dict{Tuple{Array{Symbol, 1}, Array{Symbol, 1}}, Float64} = @distributed (rd_code_pair) for comp in to_distribute
                   (partition_list, part, data_raw_code_dict[partition_list].codes, data_raw_code_dict[part].codes, n ÷ SUB_SAMPLE_DIVISOR)
                #     (partition_list, comp[1], comp[2], comp[3], n ÷ SUB_SAMPLE_DIVISOR)
                end
            end
            # for part in all_other_parts
            # for part in all_other_parts # single
            #     overlapped_ent = overlappedEnt(data_raw_code_dict[current_partition_pair].codes, data_raw_code_dict[part].codes, n)
            #     part_ent = data_raw_code_dict[part].entropy
            #     enqueue!(partition_ig_pq, (current_partition_pair, part), (overlapped_ent - data_raw_code_dict[current_partition_pair].entropy - part_ent) / (current_partition_pair_length + flattenedLength(part)))
            # end
            for (part::Tuple{Array{Symbol, 1}, Array{Symbol, 1}}, overlapped_ent::Float64) in overlapped_ent_to_add # parallel dict
                # part[1] is always current_partition_pair by construction
                enqueue!(partition_ig_pq, part, (overlapped_ent - data_raw_code_dict[part[1]].entropy - data_raw_code_dict[part[2]].entropy) / (current_partition_pair_length + length(part[2])))
            end
            # for elem in overlapped_ent_to_add # parallel array
            #     # elem[1][1] is always current_partition_pair by construction
            #     enqueue!(partition_ig_pq, elem[1], (elem[2] - data_raw_code_dict[current_partition_pair].entropy - data_raw_code_dict[elem[1][2]].entropy) / (current_partition_pair_length + flattenedLength(elem[1][2])))
            # end
end
        else
            global failsAfterSuccMerge += 1
        end
        global secMergeFailThresh = trunc(Int, length(partition_ig_pq) * SEC_MERGE_FAIL_PROP) + 1
    end
    end

    # get final scores
    code_matrix = Array{BitArray, 1}()
    for (part, code_dict) in partition_code_dict
        for (value, code_entry) in code_dict
            push!(code_matrix, code_entry.code)
        end
    end
    println("size of code_matrix ", length(code_matrix))
    code_matrix_reshape = reshape(reduce(vcat, code_matrix), (length(code_matrix), n))
    freqs = sum(code_matrix_reshape, dims = 2)
    total_freq = sum(freqs)
    pattern_costs = -log.(MI_BASE, freqs / total_freq)
    intrinsic_scores = code_matrix_reshape' * pattern_costs
end
