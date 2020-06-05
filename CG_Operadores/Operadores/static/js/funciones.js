function conversion_value_to_name(value)
{
    second_part = "";
    if (value == 1)
    {
        second_part = "data_thresholding";
    }
    else if(value == 2)
    {
        second_part = "data_contrast";
    }
    else if(value == 3)
    {
        second_part= "data_equalization";
    }
    else if(value == 4)
    {
        second_part = "data_logarithm";
    }
    else if(value == 5)
    {
        second_part = "data_square";
    }
    else if(value == 6)
    {
        second_part = "data_exponential";
    }
    else if(value == 7)
    {
        second_part = "data_pow";
    }
    else if(value == 8)
    {
        second_part = "data_addition";
    }
    else if(value == 9)
    {
        second_part = "data_difference";
    }
    else if(value == 10)
    {
        second_part = "data_dot";
    }
    else if(value == 11)
    {
        second_part = "data_division";
    }
    return second_part;
}

function erase_extension(name)
{
    val = name.split('.');
    answer = "";
    for(var i = 0; i < val.length-1; i++)
    {
        if(i != 0)
        {
            answer += "."
        }
        answer+= val[i];
    }
    return answer;
}

function get_url_to_send(name,value)
{
    first_part = erase_extension(name);
    second_part = conversion_value_to_name(value);
    return first_part+ "/" +second_part;
}