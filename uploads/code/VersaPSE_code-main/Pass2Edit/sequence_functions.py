def transformation(pwd, tar):
    """Align two keypress sequences and derive the edit-operation path.

    Uses dynamic programming to compute the edit distance, then backtracks to
    obtain the operation sequence. Operations are represented as lists:
      [1, pos, char]  -> insert 'char' at position 'pos' in the password
      [0, pos]        -> delete the character at position 'pos'
      [2, 0]          -> EOS marker
    Returns a list of such operations.
    """
    dp = []
    # Build the DP matrix for edit distance
    for i in range(len(pwd) + 1):
        dp_row = []
        if i == 0:
            for i1 in range(len(tar) + 1):
                dp_row.append(i1)
            dp.append(dp_row)
            continue
        for i1 in range(len(tar) + 1):
            if i1 == 0:
                dp_row.append(i)
            else:
                if pwd[i - 1] == tar[i1 - 1]:
                    dp_row.append(dp[i - 1][i1 - 1])
                else:
                    dp_row.append(min(dp_row[i1 - 1], dp[i - 1][i1]) + 1)
        dp.append(dp_row)

    # Backtrack from the bottom-right corner to recover the operation sequence
    psuedo_path = []
    curr = [len(pwd), len(tar)]
    while curr != [0, 0]:
        if curr[0] == 0:
            psuedo_path.insert(0, [1, curr[0], tar[curr[1] - 1]])  # 1: insertion into pwd
            curr[1] -= 1
            continue
        if curr[1] == 0:
            psuedo_path.insert(0, [0, curr[0] - 1])  # 0: deletion from pwd
            curr[0] -= 1
            continue
        if pwd[curr[0] - 1] == tar[curr[1] - 1]:
            curr[0] -= 1
            curr[1] -= 1
            continue
        else:
            if dp[curr[0]][curr[1] - 1] > dp[curr[0] - 1][curr[1]]:
                psuedo_path.insert(0, [0, curr[0] - 1])  # 0: deletion from pwd
                curr[0] -= 1
            else:
                psuedo_path.insert(0, [1, curr[0], tar[curr[1] - 1]])  # 1: insertion into pwd
                curr[1] -= 1
    psuedo_path.append([2, 0])  # 2: EOS marker

    # Convert the pseudo path into actual string positions, tracking an offset
    # that accounts for previously inserted/deleted characters. '\xde' marks a
    # deleted character that is kept in the string as a placeholder.
    path = []
    original = pwd + '\n'
    curr = pwd + '\n'
    offset = 0
    temp_offset = 0
    last_deletion = -100
    for i in psuedo_path:
        idx = i[1] + offset
        if i[0] == 0:
            if i[1] - last_deletion != 1:
                offset = max(temp_offset, offset)
                temp_offset = offset
                idx = i[1] + offset
            curr = curr[:i[1] + temp_offset] + '\xde' + curr[i[1] + temp_offset + 1:]
            idx = i[1] + temp_offset
            offset -= 1
            last_deletion = i[1]
            path.append([i[0], idx])
        elif i[0] == 1:
            if i[1] - last_deletion != 1:
                offset = max(temp_offset, offset)
                temp_offset = offset
                idx = i[1] + offset
            if curr[idx] != '\xde':
                curr = curr[:idx] + str(i[2]) + curr[idx:]
                original = original[:idx] + '\xde' + original[idx:]
                offset += 1
            else:
                curr = curr[:idx] + str(i[2]) + curr[idx + 1:]
                offset += 1
            path.append([i[0], idx, i[2]])
        else:
            path.append(i)
        while len(original) < 30:
            original += '\xde'
            curr += '\xde'
    return path


def trans2seq(path):
    """Serialize an operation list into a string form, e.g. "<1,2,a>\xa1<0,1>\xa1".

    '\xa1' separates consecutive operations and '，' (full-width comma) separates
    fields inside an operation to avoid collisions with dirty data.
    """
    editSequence = str()
    for i in path:
        currOp = '<'
        for i1 in range(len(i)):
            currOp += str(i[i1])
            if i1 != len(i) - 1:
                currOp += '，'
        currOp += '>\xa1'
        editSequence += currOp
    return editSequence


def seq2trans(seq):
    """Parse a serialized edit sequence back into an operation list."""
    path = []
    temp = seq.split('\xa1')
    if temp[-1] == '':
        del temp[-1]
    for i in temp:
        i = i.strip('<')
        i = i.strip('>')
        tempEdit = i.split('，')
        tempOperation = []
        if tempEdit[0] == '1':
            tempOperation.append(int(tempEdit[0]))
            tempOperation.append(int(tempEdit[1]))
            tempOperation.append(tempEdit[2])
        else:
            tempOperation.append(int(tempEdit[0]))
            tempOperation.append(int(tempEdit[1]))
        path.append(tempOperation)
    return path


def pass2edit(pwd, seq):
    """Apply an edit sequence step by step and record the intermediate states.

    Returns a list of triples [source_at_step, target_at_step, operation] where
    each element is one atomic edit step. '\xde' is used as a placeholder for a
    deleted character, and trailing placeholders are trimmed.
    """
    curr_list = []
    original = pwd + '\n'
    curr = pwd + '\n'
    trans = seq2trans(seq)
    for i in trans:
        step_curr = curr
        step_original = original
        if i[0] == 0:
            curr = curr[:i[1]] + '\xde' + curr[i[1] + 1:]
        elif i[0] == 1:
            if curr[i[1]] != '\xde':
                curr = curr[:i[1]] + str(i[2]) + curr[i[1]:]
                original = original[:i[1]] + '\xde' + original[i[1]:]
            else:
                curr = curr[:i[1]] + str(i[2]) + curr[i[1] + 1:]
        temp = []
        # Trim trailing placeholders so the working strings stay within MAX_LEN
        while len(curr) > 30 and original[-1] == '\xde' and curr[-1] == '\xde':
            original = original[:-1]
            curr = curr[:-1]

        temp.append(step_original)
        temp.append(step_curr)
        temp.append(i)
        curr_list.append(temp)
    return curr_list


def single_edit(original, curr, op):
    """Apply a single edit operation and return the updated strings."""
    trans = seq2trans(op)
    for i in trans:
        if i[0] == 0:
            curr = curr[:i[1]] + '\xde' + curr[i[1] + 1:]
        elif i[0] == 1:
            if curr[i[1]] != '\xde':
                curr = curr[:i[1]] + str(i[2]) + curr[i[1]:]
                original = original[:i[1]] + '\xde' + original[i[1]:]
            else:
                curr = curr[:i[1]] + str(i[2]) + curr[i[1] + 1:]
        temp = []
        while len(curr) > 30 and original[-1] == '\xde' and curr[-1] == '\xde':
            original = original[:-1]
            curr = curr[:-1]
    return original, curr, trans[0]


def applyOperation(pwd, operation):
    """Apply an operation (serialized string) to a password and return the result."""
    curr = pwd + '\n'
    trans = seq2trans(operation)
    for i in trans:
        if i[0] == 0:
            curr = curr[:i[1]] + '\xde' + curr[i[1] + 1:]
        elif i[0] == 1:
            if curr[i[1]] != '\xde':
                curr = curr[:i[1]] + str(i[2]) + curr[i[1]:]
            else:
                curr = curr[:i[1]] + str(i[2]) + curr[i[1] + 1:]
        temp = []
        while len(curr) > 30 and curr[-1] == '\xde':
            curr = curr[:-1]
    return curr.rstrip('\n')
