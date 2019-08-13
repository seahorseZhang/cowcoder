import java.util.ArrayList;
import java.util.Scanner;

//Solution to maximum distance problem from:　ByteDance
//P为给定的二维平面整数点集。定义 P 中某点x，如果x满足 P 中任意点都不在 x 的右上方区域内（横纵坐标都大于x），则称其为“最大的”。求出所有“最大的”点的集合。（
//所有点的横坐标和纵坐标都不重复, 坐标轴范围在[0, 1e9) 内）
//如下图：实心点为满足条件的点的集合。请实现代码找到集合 P 中的所有 ”最大“ 点的集合并输出。
public class MaxDistance {
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		Scanner sc = new Scanner(System.in);
		int n = sc.nextInt();
		ArrayList<Point> points = new ArrayList<>();
		ArrayList<Point> res = new ArrayList<>();
		for(int i = 0; i< n; i++) {
			Point p = new Point(sc.nextInt(), sc.nextInt());
			points.add(p);
		}
		for(Point p: points) {
			boolean temp = true;
			for(Point p1: points)
				if(p1.x > p.x && p1.y > p.y) {
					temp = false;
					break;
				}
			if(temp) res.add(p);
		}
		
		for(Point p: res) {
			System.out.printf("%d%s%d", p.x, ' ', p.y);
			System.out.println();
		}
		
	}

}

class Point {
	int x ;
	int y;
	Point(int x, int y){
		this.x = x;
		this.y = y;
	}
}
