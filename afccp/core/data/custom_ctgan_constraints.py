from sdv.constraints import create_custom_constraint_class


def is_valid_rotc(column_names, data):
    """
    Is valid function used to ensure ROTC cadets don't have USAFA pilot preferences
    """
    boolean_column = column_names[0]  # "SOC"
    numerical_column = column_names[1]  # "11XX_U_Cadet"

    # if the first column is True, the second must be 0
    true_values = (data[boolean_column] == "ROTC") & (data[numerical_column] == 0)

    # if the first is False, then the second can be anything
    false_values = (data[boolean_column] == "USAFA")

    return (true_values) | (false_values)


def is_valid_usafa(column_names, data):
    """
    Is valid function used to ensure USAFA cadets don't have ROTC pilot preferences
    """
    boolean_column = column_names[0]  # "SOC"
    numerical_column = column_names[1]  # "11XX_R_Cadet"

    # if the first column is True, the second must be 0
    true_values = (data[boolean_column] == "USAFA") & (data[numerical_column] == 0)

    # if the first is False, then the second can be anything
    false_values = (data[boolean_column] == "ROTC")

    return (true_values) | (false_values)


def transform_rotc(column_names, data):
    boolean_column = column_names[0]  # "SOC"
    numerical_column = column_names[1]  # "11XX_U_Cadet"

    # let's replace the 0 values with a typical value (median)
    typical_value = data[numerical_column].median()
    data[numerical_column] = data[numerical_column].mask(data[boolean_column] == "ROTC", typical_value)

    return data


def transform_usafa(column_names, data):
    boolean_column = column_names[0]  # "SOC"
    numerical_column = column_names[1]  # "11XX_R_Cadet"

    # let's replace the 0 values with a typical value (median)
    typical_value = data[numerical_column].median()
    data[numerical_column] = data[numerical_column].mask(data[boolean_column] == "USAFA", typical_value)

    return data


def reverse_transform_rotc(column_names, data):
    boolean_column = column_names[0]  # "SOC"
    numerical_column = column_names[1]  # "11XX_U_Cadet"

    # set the numerical column to 0 if the boolean is True
    data[numerical_column] = data[numerical_column].mask(data[boolean_column] == "ROTC", 0)

    return data


def reverse_transform_usafa(column_names, data):
    boolean_column = column_names[0]  # "SOC"
    numerical_column = column_names[1]  # "11XX_R_Cadet"

    # set the numerical column to 0 if the boolean is True
    data[numerical_column] = data[numerical_column].mask(data[boolean_column] == "USAFA", 0)

    return data

# Create the custom constraint class
IfROTCNo11XX_U = create_custom_constraint_class(
    is_valid_fn=is_valid_rotc,
    transform_fn=transform_rotc,
    reverse_transform_fn=reverse_transform_rotc
)

# Create the custom constraint class
IfUSAFANo11XX_R = create_custom_constraint_class(
    is_valid_fn=is_valid_usafa,
    transform_fn=transform_usafa,
    reverse_transform_fn=reverse_transform_usafa
)