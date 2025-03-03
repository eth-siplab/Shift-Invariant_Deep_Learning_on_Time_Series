clear all
close all
clc
%%
% Define the folder paths
currentFolder = fileparts(mfilename('fullpath'));
labelFile = fullfile(currentFolder, 'Labels.xlsx');

% Read the labels file
labels = readtable(labelFile);

% Initialize a cell array for data storage
data = cell(height(labels), 3); % Each row: {segments, COPD degree, sampling rate}

% Get all WAV files in the folder
wavFiles = dir(fullfile(currentFolder, '*.wav'));
if isempty(wavFiles)
    error('No WAV files found in the folder.');
end

% Extract unique subject IDs from the filenames
subjectIDs = unique(regexp({wavFiles.name}, 'H\d+', 'match', 'once'));

% Window and overlap durations (in seconds)
windowDuration = 8; % 8 seconds
overlapDuration = 2; % 2 seconds

% Process each unique subject
for s = 1:length(subjectIDs)
    subjectID = subjectIDs{s};
    
    % Find all files for the current subject
    subjectFiles = wavFiles(contains({wavFiles.name}, subjectID));
    
    if isempty(subjectFiles)
        warning('No files found for subject %s. Skipping.', subjectID);
        continue;
    end
    
    % Temporary storage for Left and Right channels
    leftChannels = cell(6, 1); % 6 channels for Left
    rightChannels = cell(6, 1); % 6 channels for Right
    Fs = 0; % Sampling rate placeholder
    
    % Read all files for the current subject
    for j = 1:length(subjectFiles)
        filename = subjectFiles(j).name;
        filePath = fullfile(currentFolder, filename);
        
        % Parse side and channel from filename (e.g., H002_L1.wav)
        tokens = regexp(filename, 'H\d+_([LR])(\d+)', 'tokens');
        if isempty(tokens)
            warning('Filename %s does not match expected pattern. Skipping.', filename);
            continue;
        end
        
        tokens = tokens{1}; % Extract matched tokens
        side = tokens{1}; % Left or Right
        channel = str2double(tokens{2}); % Channel number
        
        % Read the audio file
        [audioData, Fs] = audioread(filePath);
        
        % Store data in the appropriate temporary array
        if strcmp(side, 'L')
            leftChannels{channel} = audioData;
        elseif strcmp(side, 'R')
            rightChannels{channel} = audioData;
        end
    end
    
    % Trim both Left and Right channels to the shortest valid length
    validLengths = [cellfun(@length, leftChannels), cellfun(@length, rightChannels)];
    validLengths(validLengths == 0) = []; % Exclude empty channels
    if isempty(validLengths)
        warning('No valid data for subject %s. Skipping.', subjectID);
        continue;
    end
    minLength = min(min(validLengths)); % Shortest valid length
    
    % Trim all channels to the shortest valid length
    for ch = 1:6
        if ~isempty(leftChannels{ch})
            leftChannels{ch} = leftChannels{ch}(1:minLength);
        end
        if ~isempty(rightChannels{ch})
            rightChannels{ch} = rightChannels{ch}(1:minLength);
        end
    end
    
    % Combine Left and Right channels into a single matrix
    combinedData = cat(2, horzcat(leftChannels{:}), horzcat(rightChannels{:})); % Concatenate along columns

    % Segment the trimmed data into windows with overlap
    windowSize = round(windowDuration * Fs); % Calculate window size in samples
    overlapSize = round(overlapDuration * Fs); % Calculate overlap size in samples
    numSamples = size(combinedData, 1);
    numSegments = ceil((numSamples - overlapSize) / (windowSize - overlapSize));
    paddedLength = (numSegments - 1) * (windowSize - overlapSize) + windowSize;
    
    % Handle cases where data may still be insufficient for segmentation
    if numSamples < windowSize
        combinedData = [combinedData; zeros(windowSize - numSamples, size(combinedData, 2))];
    end
    
    % Buffer the data into segments
    segmentedData = zeros(windowSize, numSegments, 12); % Preallocate
    for ch = 1:12
        segmentedData(:, :, ch) = buffer(combinedData(:, ch), windowSize, overlapSize, 'nodelay');
    end
    
    % Locate the subject's row in the labels table
    labelIndex = find(strcmpi(labels.PatientID, subjectID));
    if isempty(labelIndex)
        warning('Subject %s not found in labels file. Skipping.', subjectID);
        continue;
    end
    
    % Store the segmented data and sampling rate in the data cell array
    data{labelIndex, 1} = segmentedData;
    data{labelIndex, 3} = Fs; % Store the sampling rate
end

% Extract unique COPD degree labels and assign integers
uniqueLabels = unique(labels.Diagnosis, 'stable'); % Ensure stable order
labelMap = containers.Map(uniqueLabels, 0:length(uniqueLabels)-1);

% Assign COPD degrees as integers in the cell array
for i = 1:height(labels)
    copdDegree = labels.Diagnosis{i}; % Original COPD degree
    data{i, 2} = labelMap(copdDegree); % Map to integer
end

fprintf('Data processing complete. Check "data" for segments and sampling rates.\n');
% save('respTR.mat','data')
