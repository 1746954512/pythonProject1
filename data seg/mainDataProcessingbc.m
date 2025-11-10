function mainDataProcessing()
    % =================== 输入输出目录与文件设置 ===================
    % ===== 输入文件选择 =====
    defaultInputDir = 'E:\vofa\x64\滑动输入A';
    [inputFileName, inputDir] = uigetfile('*.csv', '请选择输入CSV文件', defaultInputDir);
    if isequal(inputFileName, 0)
        disp('❌ 用户取消了文件选择。');
        return;
    end
    inputFile = fullfile(inputDir, inputFileName);
    fprintf('✅ 已选择输入文件：%s\n', inputFile);

    % ===== 输出目录选择 =====
    defaultOutputDir = 'E:\数据\第三篇\滑动输入\B';
    outputDir = uigetdir(defaultOutputDir, '请选择输出目录');
    if isequal(outputDir, 0)
        disp('❌ 用户取消了输出目录选择。');
        return;
    end
    fprintf('✅ 已选择输出目录：%s\n', outputDir);

    % 输入文件名
    outputFilePrefix = '!';                       % 输出文件名前缀

    % =================== 数据处理参数 ===================
    threshold = 1;            % 突变点检测阈值，数值越大越严格
    windowSize = 10;          % 平滑滤波窗口大小，用于数据平滑
    finalSize = [4, 100];     % 最终数据的尺寸设置 [通道数, 样本点数]
    targetChannels = finalSize(1);
    preTriggerPoints = 30;    % 突变点前保留的点数，用于调整对齐数据
    pointInterval = 10;       % 突变检测时的间隔点数
    min_mut_gap = 10;         % 突变点最小间隔，避免过于密集的突变点被判定
    saveRawSegment = false;    % 是否保存原始未处理的分割段数据

    % =================== 分割依据通道设置 ===================
    splitChannel = 3;         % 用于突变点检测的分割依据通道，设置为1代表使用第1个通道
    
    % =================== 图形显示设置 ===================
    figurePosition = [100, 100, 1200, 800];  % 图形窗口的位置和大小 [x, y, width, height]
    showGrid = true;          % 是否在图形中显示网格线
    showLegend = true;        % 是否在图形中显示图例
    channelColors = lines(targetChannels); % 为每个通道分配不同的颜色，共 4 个通道

    % =================== 回退功能设置 ===================
    maxHistory = 5;           % 最大历史记录条数
    savedFilesHistory = {};   % 保存的文件历史记录
    rawFilesHistory = {};     % 原始文件历史记录(当saveRawSegment为true时)

    % =================== 初始化 ===================
    if ~exist(outputDir, 'dir')
        mkdir(outputDir);
    end

    % 获取下一个可用文件编号
    num_files = getNextFileNumber(outputDir, outputFilePrefix);
    fprintf('✅ 下一个可用文件编号: %d\n', num_files);

    f = figure('Position', figurePosition, 'Name', '数据分割与处理实时监控 (Y保存/N跳过/Q退出/B回退)', 'NumberTitle', 'off');

    inputFile = fullfile(inputDir, inputFileName);
    if ~exist(inputFile, 'file')
        error('输入文件不存在: %s', inputFile);
    end
    all_data = readmatrix(inputFile);
    if size(all_data, 2) > 1
        all_data(:, 1) = [];
    end
    start_idx = 1;
    targetChannels = finalSize(1);
    targetSamples = finalSize(2);
    min_required_length = targetSamples * 2 + preTriggerPoints;

    % =================== 主处理循环 ===================
    while start_idx <= size(all_data, 1) - min_required_length
        currentSegment = all_data(start_idx:start_idx+targetSamples*2-1, :);
        column_1 = currentSegment(:, splitChannel);
        diff_col = abs(column_1(1:end-pointInterval) - column_1(pointInterval+1:end));
        raw_mut_points = find(diff_col > threshold);

        if isempty(raw_mut_points)
            start_idx = start_idx + 1;
            continue;
        end

        % 去除过于密集的突变点
        mut_points = raw_mut_points([true; diff(raw_mut_points) > min_mut_gap]);
        align_point = mut_points(1) - preTriggerPoints;
        valid_range = [max(1, align_point), min(size(currentSegment, 1), align_point + targetSamples - 1)];

        if valid_range(2) - valid_range(1) + 1 < targetSamples
            start_idx = start_idx + 1;
            continue;
        end

        aligned_data = currentSegment(valid_range(1):valid_range(2), :);

        % 调整通道数
        if size(aligned_data, 2) > targetChannels
            aligned_data = aligned_data(:, 1:targetChannels);
        elseif size(aligned_data, 2) < targetChannels
            aligned_data = [aligned_data, zeros(size(aligned_data, 1), targetChannels - size(aligned_data, 2))];
        end

        % 平滑与基线校正
        transposed_data = aligned_data';
        processedData = zeros(size(transposed_data));
        for i = 1:targetChannels
            smoothed = movmean(transposed_data(i, :), windowSize);
            processedData(i, :) = smoothed - smoothed(1);
        end

        % =================== 可视化 ===================
        clf(f);
        set(f, 'Name', sprintf('数据分割与处理实时监控 (Y保存/N跳过/Q退出/B回退) - 已保存%d个文件', num_files-1));
        
        subplot(3,1,1);
        hold on;
        plot(start_idx:start_idx+size(currentSegment,1)-1, currentSegment(:, 1:targetChannels));
        xline(start_idx + valid_range(1)-1, 'r-', 'Start');
        xline(start_idx + valid_range(2)-1, 'r-', 'End');
        title(sprintf('原始数据流 (当前位置: %d-%d)', start_idx, start_idx+size(currentSegment,1)-1));
        xlabel('全局样本点'); ylabel('数值');
        if showLegend
            legend(arrayfun(@(x) sprintf('通道 %d', x), 1:size(currentSegment,2), 'UniformOutput', false));
        end
        if showGrid, grid on; end

        subplot(3,1,2);
        hold on;
        for ch = 1:targetChannels
            plot(aligned_data(:, ch), 'Color', channelColors(ch,:), 'LineWidth', 1.5);
        end
        title('分割后的原始数据'); xlabel('样本点'); ylabel('数值');
        if showLegend
            legend(arrayfun(@(x) sprintf('通道 %d', x), 1:targetChannels, 'UniformOutput', false));
        end
        if showGrid, grid on; end

        subplot(3,1,3);
        hold on;
        for ch = 1:targetChannels
            plot(processedData(ch, :), 'Color', channelColors(ch,:), 'LineWidth', 1.5, 'LineStyle', '--');
        end
        title('处理后的数据 (平滑+基线校正)'); xlabel('样本点'); ylabel('数值');
        if showLegend
            legend(arrayfun(@(x) sprintf('通道 %d 处理后', x), 1:targetChannels, 'UniformOutput', false));
        end
        if showGrid, grid on; end

        % =================== 交互操作 ===================
        decision = false;
        while true
            w = waitforbuttonpress;
            if w == 1
                key = lower(get(gcf, 'CurrentCharacter'));
                switch key
                    case 'y'  % 保存当前分段
                        decision = true;
                        break;
                    case 'n'  % 跳过当前分段
                        decision = false;
                        break;
                    case 'q'  % 退出程序
                        close(f);
                        fprintf('用户中断处理\n');
                        return;
                    case 'b'  % 回退操作
                        if ~isempty(savedFilesHistory)
                            lastFile = savedFilesHistory{end};
                            if exist(lastFile, 'file')
                                delete(lastFile);
                                fprintf('已回退删除文件: %s\n', lastFile);
                            end
                            savedFilesHistory(end) = [];
                            
                            if saveRawSegment && ~isempty(rawFilesHistory)
                                lastRawFile = rawFilesHistory{end};
                                if exist(lastRawFile, 'file')
                                    delete(lastRawFile);
                                end
                                rawFilesHistory(end) = [];
                            end
                            
                            num_files = max(1, num_files - 1);
                            set(f, 'Name', sprintf('数据分割与处理实时监控 (Y保存/N跳过/Q退出/B回退) - 已保存%d个文件', num_files-1));
                            fprintf('已回退到前一个状态，当前保存文件数: %d\n', num_files-1);
                        else
                            fprintf('没有可回退的操作历史\n');
                        end
                        continue;  % 继续等待用户输入
                    otherwise
                        disp('无效输入，请按 [Y]保存 / [N]跳过 / [Q]退出 / [B]回退');
                end
            end
        end

        if ~decision
            start_idx = start_idx + valid_range(2);
            continue;
        end

        % ============ 保存数据 ============
        % 确保文件名唯一
        outputFileName = fullfile(outputDir, sprintf('%s.%d.csv', outputFilePrefix, num_files));
        while exist(outputFileName, 'file')
            num_files = num_files + 1;
            outputFileName = fullfile(outputDir, sprintf('%s.%d.csv', outputFilePrefix, num_files));
        end
        
        writematrix(processedData, outputFileName);
        
        % 添加到历史记录(保持最多maxHistory条记录)
        savedFilesHistory{end+1} = outputFileName;
        if length(savedFilesHistory) > maxHistory
            savedFilesHistory(1) = [];
        end

        if saveRawSegment
            rawFileName = fullfile(outputDir, sprintf('%s.raw.%d.csv', outputFilePrefix, num_files));
            writematrix(aligned_data', rawFileName);
            rawFilesHistory{end+1} = rawFileName;
            if length(rawFilesHistory) > maxHistory
                rawFilesHistory(1) = [];
            end
        end

        fprintf('已保存分段 %d\n', num_files);
        num_files = num_files + 1;  % 递增文件编号
        start_idx = start_idx + valid_range(2);
    end

    close(f);
    fprintf('处理完成，共保存 %d 个有效分段\n', num_files-1);
end

% =================== 辅助函数 ===================
function nextNum = getNextFileNumber(outputDir, prefix)
    % 获取目录下所有匹配 prefix.*.csv 的文件
    fileList = dir(fullfile(outputDir, [prefix '.*.csv']));
    if isempty(fileList)
        nextNum = 1;
        return;
    end
    
    % 提取所有文件编号
    fileNums = zeros(1, length(fileList));
    for i = 1:length(fileList)
        [~, name, ~] = fileparts(fileList(i).name);
        tokens = regexp(name, [prefix '\.(\d+)'], 'tokens');
        if ~isempty(tokens)
            fileNums(i) = str2double(tokens{1}{1});
        end
    end
    
    % 返回最大编号 +1
    if isempty(fileNums)
        nextNum = 1;
    else
        nextNum = max(fileNums) + 1;
    end
end