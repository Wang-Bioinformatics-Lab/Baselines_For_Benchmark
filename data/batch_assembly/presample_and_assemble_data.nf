#!/usr/bin/env nextflow
nextflow.enable.dsl=2

params.conda_path = "$baseDir/../../src/shared/conda_env/"

TOOL_FOLDER = "$baseDir/../"

params.train_path = ''
params.train_pairs_path = ''
params.tanimoto_scores_path = ''
params.batch_size = ''
params.num_turns = ''

params.save_dir = ''
params.save_format = 'hdf5'
params.num_epochs = 1800
params.low_io = false

// Filtering Parameters
params.mass_analyzer_lst = ""   // A semicolon delimited list of mass analyzers to include, an empty string means all are included

params.force_one_epoch = false

params.memory_efficent = false

split_size = 50     // How many epochs each worker has, num_epochs must be divisible by this numebr to avoid errors

// This process samples one epoch of data
process sampleSeveralEpochs {
    conda "$params.conda_path"

    errorStrategy { sleep(Math.pow(2, task.attempt) * 60 as long); return 'retry' }
    maxRetries 5

    // If not using an hdf5 pairs file, this is quite memory intensive, 
    // and you'll likely want to aggressively limit parallelism
    maxForks 10 //5

    input:
    each epoch_num 
    path train_path
    path train_pairs_path
    path tanimoto_scores_path

    output:
    path '*.hdf5'

    """
    # For each parameter, if it is not '', then we assemble it and add it to the command.
    # This will allow nextflow to work as a flexible wrapper around the python script and
    # the python script will handle any parameter errors.
    echo "params.train_path: $params.train_path"
    if [ "$params.train_path" != '' ]; then
        train_path='--train_path $train_path'
    else
        train_path=''
    fi
    echo "params.train_pairs_path: $params.train_pairs_path"
    if [ "$params.train_pairs_path" != '' ]; then
        train_pairs_path='--train_pairs_path $train_pairs_path'
    else
        train_pairs_path=''
    fi
    echo "params.tanimoto_scores_path: $params.tanimoto_scores_path"
    if [ "$params.tanimoto_scores_path" != '' ]; then
        tanimoto_scores_path='--tanimoto_scores_path $tanimoto_scores_path'
    else
        tanimoto_scores_path=''
    fi
    if [ "$params.batch_size" != '' ]; then
        batch_size='--batch_size $params.batch_size'
    else
        batch_size=''
    fi
    if [ "$params.num_turns" != '' ]; then
        num_turns='--num_turns $params.num_turns'
    else
        num_turns=''
    fi
    if [ "$params.save_format" != '' ]; then
        save_format='--save_format $params.save_format'
    else
        save_format=''
    fi
    if [ "$params.low_io" = true ]; then
        low_io='--low_io'
    else
        low_io=''
    fi
    if [ "$params.memory_efficent" = true ]; then
        memory_efficent='--memory_efficent'
    else
        memory_efficent=''
    fi
    if [ "$params.mass_analyzer_lst" != '' ]; then
        mass_analyzer_lst='--mass_analyzer_lst $params.mass_analyzer_lst'
    else
        mass_analyzer_lst=''
    fi

    # For validation, we'll want exactly one epoch
    if [ "$params.force_one_epoch" = true ]; then
        num_epochs='--num_epochs 1'
    else
        num_epochs='--num_epochs $split_size'
    fi

    # Print all parameters
    echo "train_path: \$train_path"
    echo "train_pairs_path: \$train_pairs_path"
    echo "tanimoto_scores_path: \$tanimoto_scores_path"
    echo "batch_size: \$batch_size"
    echo "num_turns: \$num_turns"
    echo "save_format: \$save_format"

    mkdir -p logs

    python $TOOL_FOLDER/presample_pairs_generic.py  \$train_path \
                                                    \$train_pairs_path \
                                                    \$tanimoto_scores_path \
                                                    \$batch_size \
                                                    \$num_turns \
                                                    \$save_format \
                                                    \$num_epochs \
                                                    \$low_io \
                                                    \$memory_efficent \
                                                    \$mass_analyzer_lst \
                                                    --save_dir "./" \
                                                    --seed $epoch_num
    """
}

// This process assembles the sampled epochs into one hdf5 file
process assembleEpochs {
    publishDir "$params.save_dir", mode: "copy"

    conda "$params.conda_path"

    input: 
    path hdf5_file, stageAs: "hdf5_file_*.hdf5"

    output:
    path "data.hdf5"

    """
    python3 $TOOL_FOLDER/batch_assembly/assemble_epochs.py --output_name data.hdf5
    """
}


workflow {
    epoch_ch = Channel.of(1..params.num_epochs/split_size)

    // For train_path, train_pairs_path, tanimoto_scores_path create a channel
    // If the value is '', create channel.empty
    if (params.train_path != '') {
        train_path_ch = Channel.fromPath(params.train_path)
    } else {
        train_path_ch = Channel.fromPath('NO_FILE')
    }
    if (params.train_pairs_path != '') {
        train_pairs_path_ch = Channel.fromPath(params.train_pairs_path)
    } else {
        train_pairs_path_ch = Channel.fromPath('NO_FILE1')
    }
    if (params.tanimoto_scores_path != '') {
        tanimoto_scores_path_ch = Channel.fromPath(params.tanimoto_scores_path)
    } else {
        tanimoto_scores_path_ch = Channel.fromPath('NO_FILE2')
    }
    
    h5_ch = sampleSeveralEpochs(epoch_ch, train_path_ch, train_pairs_path_ch, tanimoto_scores_path_ch)
    assembleEpochs(h5_ch.collect())
}