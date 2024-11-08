using CSV
using DataFrames
using MLLabelUtils
using LinearAlgebra
using SparseArrays

gege_binned = CSV.File("ENT_providers_allDiag_translated_noGrouping.csv", header=true) |> DataFrame

# diagnosis feature names with ICD in
diag_features_icd = [c for c in names(gege_binned) if occursin("ICD", c)]
# procedure feature
proc_feature = "PROC_CODE_KEY"

# all diagnosis features with ICD in are stacked and grouped into sets
gege_diag_basket = combine(groupby(stack(gege_binned[:, vcat(["CLAIM_ID"], diag_features_icd)], diag_features_icd), "CLAIM_ID"), :value => Set => :value_set)
# remove "Unknown"
gege_diag_basket[:, :value_set] = map(x -> setdiff(x, Set(["Unknown"])), gege_diag_basket[:, :value_set])
# procedure feature is stacked and grouped into sets
gege_proc_basket = combine(groupby(stack(gege_binned[:, vcat(["CLAIM_ID"], proc_feature)], proc_feature), "CLAIM_ID"), :value => Set => :value_set)

# join Diag and Proc together
gege_binned_claims = innerjoin(gege_diag_basket, gege_proc_basket, on = :CLAIM_ID, makeunique = true)
rename!(gege_binned_claims, ["CLAIM_ID", "Diag", "Proc"])

gege_binned_claims

# check
gege_diag_basket[findfirst(isequal(15292069700), gege_diag_basket[:, 1]), :]
gege_proc_basket[findfirst(isequal(15292069700), gege_proc_basket[:, 1]), :]
gege_binned_claims[findfirst(isequal(15292069700), gege_binned_claims[:, 1]), :]

# get all diagnosis labels
all_diag_labels = [s for s in setdiff(Set(stack(gege_binned[:, diag_features_icd], diag_features_icd)[:, :value]), Set(["Unknown"]))]
enc_diag = LabelEnc.NativeLabels(all_diag_labels)
# get all procedure labels
all_proc_labels = [s for s in Set(gege_binned[:, proc_feature])]
enc_proc = LabelEnc.NativeLabels(all_proc_labels)

num_of_points = size(gege_binned_claims)[1]
binarized_diag_T = sparse(zeros(length(all_diag_labels), num_of_points))
binarized_proc = sparse(zeros(num_of_points, length(all_proc_labels)))

for i in 1:num_of_points
    val_diag = [s for s in gege_binned_claims[i, :Diag]]
    lab_diag = convertlabel(LabelEnc.Indices, val_diag, enc_diag)
    for j in lab_diag
        binarized_diag_T[j, i] = 1
    end
    val_proc = [s for s in gege_binned_claims[i, :Proc]]
    lab_proc = convertlabel(LabelEnc.Indices, val_proc, enc_proc)
    for j in lab_proc
        binarized_proc[i, j] = 1
    end
end

joint_counts = binarized_diag_T * binarized_proc

diag_freq = sum(binarized_diag_T; dims = 2)
proc_freq = sum(binarized_proc; dims = 1)

@time begin
data_exp = Array{Dict, 1}(undef, num_of_points)
proc_exp = Dict()
for i in 1:num_of_points
    for j in findnz(binarized_proc[i, :])[1]
        rs_exp = 0
        idx_exp = -1
        for k in findnz(binarized_diag_T[:, i])[1]
            #rs = -log(2, joint_counts[k, j] / (diag_freq[k, 1] * proc_freq[1, j]) * num_of_points)
            rs = joint_counts[k, j] / (diag_freq[k, 1] * proc_freq[1, j])
            if rs > rs_exp
                rs_exp = rs
                idx_exp = k
            end
        end
        proc_exp[j] = (idx_exp, rs_exp)
    end
    data_exp[i] = proc_exp
end
end