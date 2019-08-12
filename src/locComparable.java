import java.util.Comparator;

public class locComparable implements Comparator{

	@Override
	public int compare(Object o1, Object o2) {
		// TODO Auto-generated method stub
		int number1 = Integer.parseInt((String) o1);
		int number2 = Integer.parseInt((String) o2);
	    int len1 = String.valueOf(number1).length();
	    int len2 = String.valueOf(number2).length();
	    if(len1<len2) number1 = number1 * 10^(len2 - len1);
	    else number2 = number2 * 10^(len1 - len2);
	    return number2 - number1;
	}

}
