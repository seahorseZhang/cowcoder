import java.util.Scanner;

public class Main{
	public static void main(String[] args){
		Scanner sc = new Scanner(System.in);
		int[] arr = new int[5];
		String str = sc.nextLine();
		for(int i = 0;i< 5; i++) {
			arr[i] = str.charAt(i) - '0';
			System.out.println(arr[i]);
		}
	}
}